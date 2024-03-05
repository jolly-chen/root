#include "RHnCUDA.h"

#include "CUDAHelpers.cuh"
#include "RCUDAContext.hxx"

#include "TError.h"
#include "TMath.h"
#include "ROOT/RVec.hxx"

#include <thrust/functional.h>
#include <array>
#include <vector>
#include <iostream>

#include <chrono>

#ifndef MIN
#define MIN(x, y) ((x < y) ? x : y)
#endif

namespace ROOT {
namespace Experimental {

////////////////////////////////////////////////////////////////////////////////
/// CUDA kernels

__device__ inline int FindFixBin(double x, const double *binEdges, int binEdgesIdx, int nBins, double xMin, double xMax)
{
   int bin;

   if (x < xMin) { // underflow
      bin = 0;
   } else if (!(x < xMax)) { // overflow  (note the way to catch NaN)
      bin = nBins + 1;
   } else {
      if (binEdgesIdx < 0) { // fixed bins
         bin = 1 + int(nBins * (x - xMin) / (xMax - xMin));
      } else { // variable bin sizes
         bin = 1 + CUDAHelpers::BinarySearch(nBins + 1, &binEdges[binEdgesIdx], x);
      }
   }

   return bin;
}

// Use Horner's method to calculate the bin in an n-Dimensional array.
template <unsigned int Dim>
__device__ inline int GetBin(size_t tid, double *binEdges, int *binEdgesIdx, int *nBinsAxis, double *xMin, double *xMax,
                             double *coords, size_t bulkSize, bool *mask)
{
   auto bin = 0;
   for (int d = Dim - 1; d >= 0; d--) {
      auto *x = &coords[d * bulkSize];
      auto binD = FindFixBin(x[tid], binEdges, binEdgesIdx[d], nBinsAxis[d] - 2, xMin[d], xMax[d]);
      mask[tid] *= binD > 0 && binD < nBinsAxis[d] - 1;

      // if (binD < 0)
      //    return -1;

      bin = bin * nBinsAxis[d] + binD;
   }

   return bin;
}

///////////////////////////////////////////
/// Device kernels for incrementing a bin.

template <typename T>
__device__ inline void AddBinContent(T *histogram, int bin, double weight)
{
   atomicAdd(&histogram[bin], (T)weight);
}

// TODO:
// template <>
// __device__ inline void AddBinContent(char *histogram, int bin, char weight)
// {
//    int newVal = histogram[bin] + int(weight);
//    if (newVal > -128 && newVal < 128) {
//       atomicExch(&histogram[bin], (char) newVal);
//       return;
//    }
//    if (newVal < -127)
//       atomicExch(&histogram[bin], (char) -127);
//    if (newVal > 127)
//       atomicExch(&histogram[bin], (char) 127);
// }

template <>
__device__ inline void AddBinContent(short *histogram, int bin, double weight)
{
   // There is no CUDA atomicCAS for short so we need to operate on integers... (Assumes little endian)
   short *addr = &histogram[bin];
   int *addrInt = (int *)((char *)addr - ((size_t)addr & 2));
   int old = *addrInt, assumed, newVal, overwrite;

   do {
      assumed = old;

      if ((size_t)addr & 2) {
         newVal = (assumed >> 16) + (int)weight; // extract short from upper 16 bits
         overwrite = assumed & 0x0000ffff;       // clear upper 16 bits
         if (newVal > -32768 && newVal < 32768)
            overwrite |= (newVal << 16); // Set upper 16 bits to newVal
         else if (newVal < -32767)
            overwrite |= 0x80010000; // Set upper 16 bits to min short (-32767)
         else
            overwrite |= 0x7fff0000; // Set upper 16 bits to max short (32767)
      } else {
         newVal = (((assumed & 0xffff) << 16) >> 16) + (int)weight; // extract short from lower 16 bits + sign extend
         overwrite = assumed & 0xffff0000;                          // clear lower 16 bits
         if (newVal > -32768 && newVal < 32768)
            overwrite |= (newVal & 0xffff); // Set lower 16 bits to newVal
         else if (newVal < -32767)
            overwrite |= 0x00008001; // Set lower 16 bits to min short (-32767)
         else
            overwrite |= 0x00007fff; // Set lower 16 bits to max short (32767)
      }

      old = atomicCAS(addrInt, assumed, overwrite);
   } while (assumed != old);
}

template <>
__device__ inline void AddBinContent(int *histogram, int bin, double weight)
{
   int old = histogram[bin], assumed;
   long newVal;

   do {
      assumed = old;
      newVal = max(long(-INT_MAX), min(assumed + long(weight), long(INT_MAX)));
      old = atomicCAS(&histogram[bin], assumed, newVal);
   } while (assumed != old); // Repeat on failure/when the bin was already updated by another thread
}

///////////////////////////////////////////
/// Histogram filling kernels

template <typename T, unsigned int Dim>
__global__ void HistogramLocal(T *histogram, double *binEdges, int *binEdgesIdx, int *nBinsAxis, double *xMin,
                               double *xMax, double *coords, double *weights, bool *mask, size_t nBins, size_t bulkSize)
{
   auto sMem = CUDAHelpers::shared_memory_proxy<T>();
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int localTid = threadIdx.x;
   unsigned int stride = blockDim.x * gridDim.x; // total number of threads

   // Initialize a local per-block histogram
   for (auto i = localTid; i < nBins; i += blockDim.x) {
      sMem[i] = 0;
   }
   __syncthreads();

   // Fill local histogram
   for (auto i = tid; i < bulkSize; i += stride) {
      auto bin = GetBin<Dim>(i, binEdges, binEdgesIdx, nBinsAxis, xMin, xMax, coords, bulkSize, mask);
      // if (bin >= 0)
      AddBinContent<T>(sMem, bin, weights[i]);
   }
   __syncthreads();

   // Merge results in global histogram
   for (auto i = localTid; i < nBins; i += blockDim.x) {
      AddBinContent<T>(histogram, i, sMem[i]);
   }
}

// Slower histogramming, but requires less memory.
// OPTIMIZATION: consider sorting the coords array.
template <typename T, unsigned int Dim>
__global__ void HistogramGlobal(T *histogram, double *binEdges, int *binEdgesIdx, int *nBinsAxis, double *xMin,
                                double *xMax, double *coords, double *weights, bool *mask, size_t bulkSize)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   // Fill histogram
   for (auto i = tid; i < bulkSize; i += stride) {
      auto bin = GetBin<Dim>(i, binEdges, binEdgesIdx, nBinsAxis, xMin, xMax, coords, bulkSize, mask);
      // if (bin >= 0)
         AddBinContent<T>(histogram, bin, weights[i]);
   }
}

///////////////////////////////////////////
/// Statistics calculation kernels

// Nullify weights of under/overflow mask to exclude them from stats
template <unsigned int Dim, unsigned int BlockSize>
__global__ void ExcludeUOverflowKernel(bool *mask, double *weights, size_t bulkSize)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   for (auto i = tid; i < bulkSize; i += stride) {
      weights[i] *= mask[i];
   }
}

///////////////////////////////////////////
/// RHnCUDA

template <typename T, unsigned int Dim, unsigned int BlockSize>
RHnCUDA<T, Dim, BlockSize>::RHnCUDA(std::size_t maxBulkSize, const std::size_t nBins,
                                    const std::array<int, Dim> &nBinsAxis, const std::array<double, Dim> &xLow,
                                    const std::array<double, Dim> &xHigh, const std::vector<double> &binEdges,
                                    const std::array<int, Dim> &binEdgesIdx)
   : kStatsSmemSize((BlockSize <= 32) ? 2 * BlockSize * sizeof(double) : BlockSize * sizeof(double)),
     fCtx(RCUDAContext::instance())
{
   fMaxBulkSize = maxBulkSize;
   fNBins = nBins;
   fEntries = 0;

   // Setup device memory for filling the histogram.
   ERRCHECK(cudaMalloc((void **)&fDCoords, Dim * fMaxBulkSize * sizeof(double)));
   ERRCHECK(cudaMalloc((void **)&fDWeights, fMaxBulkSize * sizeof(double)));
   ERRCHECK(cudaMalloc((void **)&fDMask, Dim * fMaxBulkSize * sizeof(int)));

   // Setup device memory for histogram characteristics
   ERRCHECK(cudaMalloc((void **)&fDNBinsAxis, Dim * sizeof(int)));
   ERRCHECK(cudaMemcpy(fDNBinsAxis, nBinsAxis.data(), Dim * sizeof(int), cudaMemcpyHostToDevice));
   ERRCHECK(cudaMalloc((void **)&fDMin, Dim * sizeof(double)));
   ERRCHECK(cudaMemcpy(fDMin, xLow.data(), Dim * sizeof(double), cudaMemcpyHostToDevice));
   ERRCHECK(cudaMalloc((void **)&fDMax, Dim * sizeof(double)));
   ERRCHECK(cudaMemcpy(fDMax, xHigh.data(), Dim * sizeof(double), cudaMemcpyHostToDevice));
   ERRCHECK(cudaMalloc((void **)&fDBinEdgesIdx, Dim * sizeof(int)));
   ERRCHECK(cudaMemcpy(fDBinEdgesIdx, binEdgesIdx.data(), Dim * sizeof(int), cudaMemcpyHostToDevice));

   fDBinEdges = NULL;
   if (binEdges.size() > 0) {
      ERRCHECK(cudaMalloc((void **)&fDBinEdges, binEdges.size() * sizeof(double)));
      ERRCHECK(cudaMemcpy(fDBinEdges, binEdges.data(), binEdges.size() * sizeof(double), cudaMemcpyHostToDevice));
   }

   // Allocate and initialize device memory for the histogram and statistics.
   ERRCHECK(cudaMalloc((void **)&fDHistogram, fNBins * sizeof(T)));
   ERRCHECK(cudaMemset(fDHistogram, 0, fNBins * sizeof(T)));
   ERRCHECK(cudaMalloc((void **)&fDStats, kNStats * sizeof(double)));
   ERRCHECK(cudaMemset(fDStats, 0, kNStats * sizeof(double)));
   ERRCHECK(cudaMalloc((void **)&fDIntermediateStats, kMaxBlocks * kNStats * sizeof(double)));

   fHistoSmemSize = fNBins * sizeof(T);

   // TODO: Allocate and transfer together?
   ERRCHECK(cudaMallocHost((void **)&fHCoords, Dim * maxBulkSize * sizeof(double)));
   ERRCHECK(cudaMallocHost((void **)&fHWeights, maxBulkSize * sizeof(double)));

   if (getenv("DBG")) {
      printf("Maximum shared memory size: %d\n", fCtx.GetMaxSMem());
   }
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
RHnCUDA<T, Dim, BlockSize>::~RHnCUDA()
{
   ERRCHECK(cudaFree(fDHistogram));
   ERRCHECK(cudaFree(fDNBinsAxis));
   ERRCHECK(cudaFree(fDMin));
   ERRCHECK(cudaFree(fDMax));
   ERRCHECK(cudaFree(fDBinEdgesIdx));
   ERRCHECK(cudaFree(fDCoords));
   ERRCHECK(cudaFree(fDWeights));
   ERRCHECK(cudaFree(fDMask));
   ERRCHECK(cudaFree(fDStats));
   ERRCHECK(cudaFree(fDIntermediateStats));
   if (fDBinEdges != NULL) {
      ERRCHECK(cudaFree(fDBinEdges));
   }
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::Fill(int numStream, const RVecD &coords)
{
   RVecD weights(coords.size() / Dim, 1);
   Fill(numStream, coords, weights);
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::Fill(int numStream, const RVecD &coords, const RVecD &weights)
{
   auto bulkSize = weights.size();
   auto stream = fCtx.GetStream(numStream);

   cudaStreamSynchronize(stream); // TODO: remove need for synchronization
   std::memcpy(fHCoords, coords.data(), Dim * bulkSize * sizeof(double));
   std::memcpy(fHWeights, weights.data(), bulkSize * sizeof(double));
   ERRCHECK(cudaMemcpyAsync(fDCoords, fHCoords, Dim * bulkSize * sizeof(double), cudaMemcpyHostToDevice, stream));
   ERRCHECK(cudaMemcpyAsync(fDWeights, fHWeights, bulkSize * sizeof(double), cudaMemcpyHostToDevice, stream));
   ERRCHECK(cudaMemsetAsync(fDMask, 1, bulkSize * sizeof(bool), stream));

   fEntries += bulkSize;
   ExecuteCUDAHisto(stream, bulkSize);
}

unsigned int nextPow2(unsigned int x)
{
   --x;
   x |= x >> 1;
   x |= x >> 2;
   x |= x >> 4;
   x |= x >> 8;
   x |= x >> 16;
   return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction
// kernel. We set threads / block to the minimum of maxThreads and n/2.
// We observe the maximum specified number of blocks, because
// each thread in that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::GetNumBlocksAndThreads(std::size_t n, int &blocks, int &threads)
{
   // get device capability, to avoid block/grid size exceed the upper bound
   threads = (n < BlockSize * 2) ? nextPow2((n + 1) / 2) : BlockSize;
   blocks = (n + (threads * 2 - 1)) / (threads * 2);

   if ((float)threads * blocks > (float)fCtx.GetMaxBlockDimX() * fCtx.GetMaxThreadsPerBlock()) {
      printf("n is too large, please choose a smaller number!\n");
   }

   if (blocks > fCtx.GetMaxBlockDimX()) {
      printf("Grid size <%d> exceeds the device capability <%d>, set block size as "
             "%d (original %d)\n",
             blocks, fCtx.GetMaxBlockDimX(), threads * 2, threads);

      blocks /= 2;
      threads *= 2;
   }

   blocks = MIN(kMaxBlocks, blocks);
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::GetStats(cudaStream_t &stream, std::size_t size)
{
   ExcludeUOverflowKernel<Dim, BlockSize><<<fmax(1, size / BlockSize), BlockSize, 0, stream>>>(fDMask, fDWeights, size);
   ERRCHECK(cudaPeekAtLastError());

   // Number of blocks in grid is halved, because each thread loads two elements from global memory.
   int numBlocks, numThreads;
   GetNumBlocksAndThreads(size, numBlocks, numThreads);

   double *resultArray;
   bool overwrite;
   if (numBlocks == 1) {
      resultArray = fDStats;
      overwrite = false;
   } else {
      resultArray = fDIntermediateStats;
      overwrite = true;
   }

   // OPTIMIZATION: interleave/change order of computation of different stats? or parallelize via
   // streams. Need to profile first.
   CUDAHelpers::TransformReduce(stream, numBlocks, numThreads, size, resultArray, 0., overwrite,
                                CUDAHelpers::Plus<double>(), CUDAHelpers::Identity{}, fDWeights);
   ERRCHECK(cudaPeekAtLastError());
   CUDAHelpers::TransformReduce(stream, numBlocks, numThreads, size, &resultArray[numBlocks], 0., overwrite,
                                CUDAHelpers::Plus<double>(), CUDAHelpers::Square{}, fDWeights);
   ERRCHECK(cudaPeekAtLastError());

   auto offset = 2 * numBlocks;
   for (auto d = 0U; d < Dim; d++) {
      // Multiply weight with coordinate of current axis. E.g., for Dim = 2 this computes Tsumwx and Tsumwy
      CUDAHelpers::TransformReduce(stream, numBlocks, numThreads, size, &resultArray[offset], 0., overwrite,
                                   CUDAHelpers::Plus<double>(), CUDAHelpers::Mul{}, fDWeights, &fDCoords[d * size]);
      ERRCHECK(cudaPeekAtLastError());
      offset += numBlocks;

      // Squares coodinate per axis. E.g., for Dim = 2 this computes Tsumwx2 and Tsumwy2
      CUDAHelpers::TransformReduce(stream, numBlocks, numThreads, size, &resultArray[offset], 0., overwrite,
                                   CUDAHelpers::Plus<double>(), CUDAHelpers::MulSquare{}, fDWeights,
                                   &fDCoords[d * size]);
      ERRCHECK(cudaPeekAtLastError());
      offset += numBlocks;

      for (auto prev_d = 0U; prev_d < d; prev_d++) {
         // Multiplies coordinate of current axis with the "previous" axis. E.g., for Dim = 2 this computes Tsumwxy
         CUDAHelpers::TransformReduce(stream, numBlocks, numThreads, size, &resultArray[offset], 0., overwrite,
                                      CUDAHelpers::Plus<double>(), CUDAHelpers::Mul3{}, fDWeights,
                                      &fDCoords[prev_d * size], &fDCoords[d * size]);
         ERRCHECK(cudaPeekAtLastError());
         offset += numBlocks;
      }
   }

   int s = numBlocks;
   int next_s;
   double *dest = fDIntermediateStats;
   while (s > 1) {
      // fDintermediateStats stores the result of the sum for each block, per statistic. We need to perform another
      // reduction to merge the per-block sums to get the total sum for each statistic.
      int threads = 0, blocks = 0;
      GetNumBlocksAndThreads(s, blocks, threads);
      next_s = (s + (threads * 2 - 1)) / (threads * 2);
      if (next_s == 1) {
         dest = fDStats;
         overwrite = false;
      }

      ERRCHECK(cudaMemcpy(fDIntermediateStats, fDStats, s * kNStats * sizeof(double), cudaMemcpyDeviceToDevice));
      for (auto i = 0U; i < kNStats; i++) {
         CUDAHelpers::TransformReduce(stream, blocks, threads, s, &dest[i * next_s], 0., overwrite,
                                      CUDAHelpers::Plus<double>(), CUDAHelpers::Identity{},
                                      &fDIntermediateStats[i * s]);
         ERRCHECK(cudaPeekAtLastError());
      }

      s = next_s;
   }
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::ExecuteCUDAHisto(cudaStream_t &stream, std::size_t size)
{
   int numBlocks = size % BlockSize == 0 ? size / BlockSize : size / BlockSize + 1;

   if (fHistoSmemSize > fCtx.GetMaxSMem()) {
      HistogramGlobal<T, Dim><<<numBlocks, BlockSize, 0, stream>>>(fDHistogram, fDBinEdges, fDBinEdgesIdx, fDNBinsAxis,
                                                                   fDMin, fDMax, fDCoords, fDWeights, fDMask, size);
   } else {
      HistogramLocal<T, Dim><<<numBlocks, BlockSize, fHistoSmemSize, stream>>>(
         fDHistogram, fDBinEdges, fDBinEdgesIdx, fDNBinsAxis, fDMin, fDMax, fDCoords, fDWeights, fDMask, fNBins, size);
   }
   ERRCHECK(cudaPeekAtLastError());

   GetStats(stream, size);
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::RetrieveResults(int numStream, T *histResult, double *statsResult)
{
   auto stream = fCtx.GetStream(numStream);

   // Copy back results from GPU to CPU.
   ERRCHECK(cudaMemcpyAsync(histResult, fDHistogram, fNBins * sizeof(T), cudaMemcpyDeviceToHost, stream));
   ERRCHECK(cudaMemcpyAsync(statsResult, fDStats, kNStats * sizeof(double), cudaMemcpyDeviceToHost, stream));
   cudaStreamSynchronize(stream);
}

#include "RHnCUDA-impl.cu"
} // namespace Experimental
} // namespace ROOT
