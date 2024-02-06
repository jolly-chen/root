#include "RHnCUDA.h"

#include "CUDAHelpers.cuh"
#include "TError.h"
#include "TMath.h"
#include "ROOT/RVec.hxx"

#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/device_free.h>

#include <array>
#include <vector>
#include <iostream>

namespace ROOT {
namespace Experimental {

template <typename T, unsigned int Dim, unsigned int BlockSize>
struct RHnCUDA<T, Dim, BlockSize>::DevicePointers {
   thrust::device_ptr<T> fHistogram; ///< Pointer to histogram buffer on the GPU.

   thrust::device_ptr<int> fNBinsAxis;   ///< Number of mask(1D) WITH u/overflow per axis
   thrust::device_ptr<double> fMin;      ///< Low edge of first bin per axis
   thrust::device_ptr<double> fMax;      ///< Upper edge of last bin per axis
   thrust::device_ptr<double> fBinEdges; ///< Bin edges array for each axis
   thrust::device_ptr<int> fBinEdgesIdx; ///< Start index of the binedges in kBinEdges per axis

   thrust::device_ptr<double> fCoords;  ///< Pointer to array of coordinates to fill on the GPU.
   thrust::device_ptr<double> fWeights; ///< Pointer to array of weights on the GPU.
   thrust::device_ptr<bool> fMask;      ///< Mask for under/overflow

   thrust::device_ptr<double> fIntermediateStats; ///< Buffer for storing intermediate results of stat reduction on GPU.
   thrust::device_ptr<double> fStats;             ///< Pointer to statistics array on GPU

   ~DevicePointers() {
      thrust::device_free(fHistogram);
      thrust::device_free(fNBinsAxis);
      thrust::device_free(fMin);
      thrust::device_free(fMax);
      thrust::device_free(fBinEdgesIdx);
      thrust::device_free(fCoords);
      thrust::device_free(fWeights);
      thrust::device_free(fMask);
      thrust::device_free(fIntermediateStats);
      thrust::device_free(fStats);
      if (fBinEdges != NULL) thrust::device_free(fBinEdges);
   }
};

////////////////////////////////////////////////////////////////////////////////
/// CUDA kernels

__device__ inline int FindFixBin(double x, const double *binEdges, int binEdgesIdx, int nBins, double xMin, double xMax)
{
   int bin;

   // OPTIMIZATION: can this be done with less branching?
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
      // if (binD < 0) {
      //    return -1;
      // }
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

// ///////////////////////////////////////////
// /// Histogram filling kernels

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

// ///////////////////////////////////////////
// /// Statistics calculation operations

struct XY {
   // Tuple: weights, mask
   __host__ __device__ double operator()(const thrust::tuple<double, double> tup) const
   {
      return thrust::get<0>(tup) * thrust::get<1>(tup);
   }
};

struct XY2 {
   // Tuple: weights, coords
   __host__ __device__ double operator()(const thrust::tuple<double, double> tup) const
   {
      return thrust::get<0>(tup) * thrust::get<1>(tup) * thrust::get<1>(tup);
   }
};

struct XYZ {
   // Tuple: weights, coords, prev_coords
   __host__ __device__ double operator()(const thrust::tuple<double, double, double> tup) const
   {
      return thrust::get<0>(tup) * thrust::get<1>(tup) * thrust::get<2>(tup);
   }
};

///////////////////////////////////////////
/// RHnCUDA

template <typename T, unsigned int Dim, unsigned int BlockSize>
RHnCUDA<T, Dim, BlockSize>::RHnCUDA(std::size_t maxBulkSize, const std::size_t nBins,
                                    const std::array<int, Dim> &nBinsAxis, const std::array<double, Dim> &xLow,
                                    const std::array<double, Dim> &xHigh, const std::vector<double> &binEdges,
                                    const std::array<int, Dim> &binEdgesIdx)
   : kStatsSmemSize((BlockSize <= 32) ? 2 * BlockSize * sizeof(double) : BlockSize * sizeof(double)), fStats(kNStats, 0)
{
   fMaxBulkSize = maxBulkSize;
   fNBins = nBins;
   fEntries = 0;
   devPtrs = std::make_unique<DevicePointers>();

   // Setup device memory for filling the histogram.
   devPtrs->fCoords = thrust::device_malloc<double>(Dim * fMaxBulkSize);
   devPtrs->fWeights = thrust::device_malloc<double>(fMaxBulkSize);
   devPtrs->fMask = thrust::device_malloc<bool>(fMaxBulkSize);

   devPtrs->fNBinsAxis = thrust::device_malloc<int>(nBinsAxis.size());
   devPtrs->fMin = thrust::device_malloc<double>(xLow.size());
   devPtrs->fMax = thrust::device_malloc<double>(xHigh.size());
   devPtrs->fBinEdgesIdx = thrust::device_malloc<int>(binEdgesIdx.size());
   thrust::copy(nBinsAxis.begin(), nBinsAxis.end(), devPtrs->fNBinsAxis);
   thrust::copy(xLow.begin(), xLow.end(), devPtrs->fMin);
   thrust::copy(xHigh.begin(), xHigh.end(), devPtrs->fMax);
   thrust::copy(binEdgesIdx.begin(), binEdgesIdx.end(), devPtrs->fBinEdgesIdx);

   devPtrs->fBinEdges = NULL;
   if (binEdges.size() > 0) {
      devPtrs->fBinEdges = thrust::device_malloc<double>(binEdges.size());
      thrust::copy(binEdges.begin(), binEdges.end(), devPtrs->fBinEdges);
   }

   // Allocate and initialize device memory for the histogram and statistics.
   devPtrs->fHistogram = thrust::device_malloc<T>(fNBins);
   // devPtrs->fStats = thrust::device_malloc<double>(kNStats);
   // devPtrs->fIntermediateStats = thrust::device_malloc<double>(kNStats);
   thrust::fill(devPtrs->fHistogram, devPtrs->fHistogram + fNBins, (T)0);
   // thrust::fill(devPtrs->fStats, devPtrs->fStats + kNStats, (double)0);
   // thrust::fill(devPtrs->fIntermediateStats, devPtrs->fIntermediateStats + kNStats, (double)0);

   cudaDeviceProp prop;
   ERRCHECK(cudaGetDeviceProperties(&prop, 0));
   fMaxSmemSize = prop.sharedMemPerBlock;
   fHistoSmemSize = fNBins * sizeof(T);

   if (getenv("DBG")) {
      printf("Maximum shared memory size: %zu\n", fMaxSmemSize);
   }
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::Fill(const RVecD &coords)
{
   RVecD weights(coords.size() / Dim, 1);
   Fill(coords, weights);
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::Fill(const RVecD &coords, const RVecD &weights)
{
   auto bulkSize = weights.size();

   thrust::copy(coords.begin(), coords.end(), devPtrs->fCoords);
   thrust::copy(weights.begin(), weights.end(), devPtrs->fWeights);
   thrust::fill(devPtrs->fMask, devPtrs->fMask + fMaxBulkSize, true);

   fEntries += bulkSize;
   ExecuteCUDAHisto(bulkSize);
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

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::GetStats(std::size_t size)
{
   auto weightsBinsB = thrust::make_zip_iterator(thrust::make_tuple(devPtrs->fWeights, devPtrs->fMask));
   auto weightsBinsE = thrust::make_zip_iterator(thrust::make_tuple(devPtrs->fWeights + size, devPtrs->fMask + size));
   thrust::transform(thrust::device, weightsBinsB, weightsBinsE, devPtrs->fWeights, XY{});

   fStats[0] +=
      thrust::reduce(thrust::device, devPtrs->fWeights, devPtrs->fWeights + size, 0., thrust::plus<double>());
   fStats[1] += thrust::transform_reduce(thrust::device, devPtrs->fWeights, devPtrs->fWeights + size,
                                         thrust::square<double>(), 0., thrust::plus<double>());

   auto offset = 2;
   for (auto d = 0U; d < Dim; d++) {
      // Multiply weight with coordinate of current axis. E.g., for Dim = 2 this computes Tsumwx and Tsumwy
      auto coordsWeightsB =
         thrust::make_zip_iterator(thrust::make_tuple(devPtrs->fWeights, devPtrs->fCoords + d * size));
      auto coordsWeightsE =
         thrust::make_zip_iterator(thrust::make_tuple(devPtrs->fWeights + size, devPtrs->fCoords + (d + 1) * size));

      fStats[offset] += thrust::transform_reduce(thrust::device, coordsWeightsB, coordsWeightsE, XY{}, 0.,
                                                 thrust::plus<double>());
      offset++;
      fStats[offset] += thrust::transform_reduce(thrust::device, coordsWeightsB, coordsWeightsE, XY2{}, 0.,
                                                 thrust::plus<double>());
      offset++;

      for (auto prev_d = 0U; prev_d < d; prev_d++) {
         // Multiplies coordinate of current axis with the "previous" axis. E.g., for Dim = 2 this computes
         // Tsumwxy
         auto prevCoordsWeightsB = thrust::make_zip_iterator(
            thrust::make_tuple(devPtrs->fWeights, devPtrs->fCoords + prev_d * size, devPtrs->fCoords + d * size));
         auto prevCoordsWeightsE = thrust::make_zip_iterator(thrust::make_tuple(
            devPtrs->fWeights + size, devPtrs->fCoords + (prev_d + 1) * size, devPtrs->fCoords + (d + 1) * size));
         fStats[offset] += thrust::transform_reduce(thrust::device, prevCoordsWeightsB, prevCoordsWeightsE, XYZ{},
                                                    0., thrust::plus<double>());
         offset++;
      }
   }
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::ExecuteCUDAHisto(std::size_t size)
{
   int numBlocks = size % BlockSize == 0 ? size / BlockSize : size / BlockSize + 1;

   if (fHistoSmemSize > fMaxSmemSize) {
      HistogramGlobal<T, Dim>
         <<<numBlocks, BlockSize>>>(devPtrs->fHistogram.get(), devPtrs->fBinEdges.get(), devPtrs->fBinEdgesIdx.get(),
                                    devPtrs->fNBinsAxis.get(), devPtrs->fMin.get(), devPtrs->fMax.get(),
                                    devPtrs->fCoords.get(), devPtrs->fWeights.get(), devPtrs->fMask.get(), size);
   } else {
      HistogramLocal<T, Dim><<<numBlocks, BlockSize, fHistoSmemSize>>>(
         devPtrs->fHistogram.get(), devPtrs->fBinEdges.get(), devPtrs->fBinEdgesIdx.get(), devPtrs->fNBinsAxis.get(),
         devPtrs->fMin.get(), devPtrs->fMax.get(), devPtrs->fCoords.get(), devPtrs->fWeights.get(),
         devPtrs->fMask.get(), fNBins, size);
   }
   ERRCHECK(cudaPeekAtLastError());

   GetStats(size);
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnCUDA<T, Dim, BlockSize>::RetrieveResults(T *histResult, double *statsResult)
{
   ERRCHECK(cudaMemcpy(histResult, devPtrs->fHistogram.get(), fNBins * sizeof(T), cudaMemcpyDeviceToHost));
   std::copy(fStats.begin(), fStats.end(), statsResult);
}

#include "RHnCUDA-impl.cu"
} // namespace Experimental
} // namespace ROOT
