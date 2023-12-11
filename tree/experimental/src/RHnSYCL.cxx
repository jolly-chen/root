#include "RHnSYCL.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <array>

#include "TMath.h"
#include "SYCLHelpers.h"
#include "ROOT/RVec.hxx"

namespace ROOT {
namespace Experimental {

using mode = sycl::access::mode;

template <class T>
using AccR = sycl::accessor<T, 1, mode::read>;
template <class T>
using AccW = sycl::accessor<T, 1, mode::write>;
template <class T>
using AccRW = sycl::accessor<T, 1, mode::read_write>;

template <class T>
using AccLM = sycl::local_accessor<T, 1>;

////////////////////////////////////////////////////////////////////////////////
/// Bin calculation methods

inline int FindFixBin(const double x, const double *binEdges, const int binEdgesIdx, const int nBins, const double xMin,
                      const double xMax)
{
   int bin;

   // OPTIMIZATION: can this be done with less branching?
   if (x < xMin) { // underflow
      bin = 0;
   } else if (!(x < xMax)) { // overflow  (note the way to catch NaN)
      bin = nBins + 1;
   } else {
      if (binEdgesIdx < 0) { // fix bins
         bin = 1 + int(nBins * (x - xMin) / (xMax - xMin));
      } else { // variable bin sizes
         bin = 1 + SYCLHelpers::BinarySearch(nBins + 1, &binEdges[binEdgesIdx], x);
      }
   }

   return bin;
}

template <unsigned int Dim>
inline int GetBin(const size_t tid, const double *binEdges, const AccR<int> &binEdgesIdx, const AccR<int> &nBinsAxis,
                  const AccR<double> &xMin, const AccR<double> &xMax, const AccR<double> &coords, const size_t bulkSize,
                  const AccW<int> &bins)
{
   auto bin = 0;
   for (int d = Dim - 1; d >= 0; d--) {
      auto *x = &coords[d * bulkSize];
      auto binD = FindFixBin(x[tid], binEdges, binEdgesIdx[d], nBinsAxis[d] - 2, xMin[d], xMax[d]);
      bins[d * bulkSize + tid] = binD;

      if (binD < 0) {
         return -1;
      }

      bin = bin * nBinsAxis[d] + binD;
   }

   return bin;
}

///////////////////////////////////////////
/// Methods for incrementing a bin.

template <sycl::access::address_space Space>
inline void AddBinContent(const AccRW<double> &histogram, int bin, double weight)
{
   auto atomic =
      sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
   atomic.fetch_add(weight);
}

template <sycl::access::address_space Space>
inline void AddBinContent(const AccRW<float> &histogram, int bin, double weight)
{
   auto atomic =
      sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
   atomic.fetch_add((float)weight);
}

template <sycl::access::address_space Space>
inline void AddBinContent(const AccRW<short> &histogram, int bin, double weight)
{
   // There is no fetch_add for short so we need to operate on integers... (Assumes little endian)
   short *addr = &histogram.get_pointer()[bin];
   int *addrInt = (int *)((char *)addr - ((size_t)addr & 2));
   int assumed, newVal, overwrite;
   bool success = false;

   do {
      assumed = *addrInt;

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

      auto atomic = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(addrInt[0]);
      success = atomic.compare_exchange_strong(assumed, overwrite);
   } while (!success);
}

template <sycl::access::address_space Space>
inline void AddBinContent(const AccRW<int> &histogram, int bin, double weight)
{
   int assumed;
   long newVal;
   bool success = false;

   // Repeat on failure/when the bin was already updated by another thread
   do {
      assumed = histogram[bin];
      newVal = sycl::max(long(-INT_MAX), sycl::min(assumed + long(weight), long(INT_MAX)));
      auto atomic =
         sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
      success = atomic.compare_exchange_strong(assumed, (int)newVal);
   } while (!success);
}

//
// TODO: Cleaner overloads for local accessors with less duplication
//

template <sycl::access::address_space Space>
inline void AddBinContent(const AccLM<double> &histogram, int bin, double weight)
{
   auto atomic =
      sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
   atomic.fetch_add(weight);
}

template <sycl::access::address_space Space>
inline void AddBinContent(const AccLM<float> &histogram, int bin, double weight)
{
   auto atomic =
      sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
   atomic.fetch_add((float)weight);
}

template <sycl::access::address_space Space>
inline void AddBinContent(const AccLM<short> &histogram, int bin, double weight)
{
   // There is no fetch_add for short so we need to operate on integers... (Assumes little endian)
   short *addr = &histogram.get_pointer()[bin];
   int *addrInt = (int *)((char *)addr - ((size_t)addr & 2));
   int assumed, newVal, overwrite;
   bool success = false;

   do {
      assumed = *addrInt;

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

      auto atomic = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(addrInt[0]);
      success = atomic.compare_exchange_strong(assumed, overwrite);
   } while (!success);
}

template <sycl::access::address_space Space>
inline void AddBinContent(const AccLM<int> &histogram, int bin, double weight)
{
   int assumed;
   long newVal;
   bool success = false;

   // Repeat on failure/when the bin was already updated by another thread
   do {
      assumed = histogram[bin];
      newVal = sycl::max(long(-INT_MAX), sycl::min(assumed + long(weight), long(INT_MAX)));
      auto atomic =
         sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
      success = atomic.compare_exchange_strong(assumed, (int)newVal);
   } while (!success);
}

///////////////////////////////////////////
/// Histogram filling kernels

template <typename T, unsigned int Dim>
class HistogramGlobal {
private:
   const AccRW<T> histogram;
   const double *binEdges;
   const AccR<double> xMin;
   const AccR<double> xMax;
   const AccR<double> coords;
   const AccR<double> weights;
   const AccR<int> binEdgesIdx;
   const AccR<int> nBinsAxis;
   const AccW<int> bins;
   const std::size_t bulkSize;

public:
   HistogramGlobal(const AccRW<T> &_histogram, const double *_binEdges, const AccR<int> &_binEdgesIdx,
                   const AccR<int> &_nBinsAxis, const AccR<double> &_xMin, const AccR<double> &_xMax,
                   const AccR<double> &_coords, const AccR<double> &_weights, const AccW<int> &_bins,
                   const std::size_t _bulkSize)
      : histogram(_histogram),
        binEdges(_binEdges),
        xMin(_xMin),
        xMax(_xMax),
        coords(_coords),
        weights(_weights),
        binEdgesIdx(_binEdgesIdx),
        nBinsAxis(_nBinsAxis),
        bins(_bins),
        bulkSize(_bulkSize)
   {
   }

   void operator()(sycl::item<1> item) const
   {
      size_t id = item.get_linear_id();
      auto bin = GetBin<Dim>(id, binEdges, binEdgesIdx, nBinsAxis, xMin, xMax, coords, bulkSize, bins);

      if (bin >= 0) {
         AddBinContent<sycl::access::address_space::global_space>(histogram, bin, weights[id]);
      }
   }
};

template <typename T, unsigned int Dim>
class HistogramLocal {
private:
   const AccLM<T> localMem;
   const AccRW<T> histogram;
   const double *binEdges;
   const AccR<double> xMin;
   const AccR<double> xMax;
   const AccR<double> coords;
   const AccR<double> weights;
   const AccR<int> binEdgesIdx;
   const AccR<int> nBinsAxis;
   const AccW<int> bins;
   const std::size_t nBins;
   const std::size_t bulkSize;

public:
   HistogramLocal(const AccLM<T> &_localMem, const AccRW<T> &_histogram, double *_binEdges,
                  const AccR<int> &_binEdgesIdx, const AccR<int> &_nBinsAxis, const AccR<double> &_xMin,
                  const AccR<double> &_xMax, const AccR<double> &_coords, const AccR<double> &_weights,
                  const AccW<int> &_bins, size_t _nBins, std::size_t _bulkSize)
      : localMem(_localMem),
        histogram(_histogram),
        binEdges(_binEdges),
        xMin(_xMin),
        xMax(_xMax),
        coords(_coords),
        weights(_weights),
        binEdgesIdx(_binEdgesIdx),
        nBinsAxis(_nBinsAxis),
        bins(_bins),
        nBins(_nBins),
        bulkSize(_bulkSize)
   {
   }

   void operator()(sycl::nd_item<1> item) const
   {
      auto globalId = item.get_global_id(0);
      auto localId = item.get_local_id(0);
      auto group = item.get_group();
      auto groupSize = item.get_local_range(0);
      auto stride = groupSize * item.get_group_range(0);

      // Initialize a local per-work-group histogram
      for (auto i = localId; i < nBins; i += groupSize) {
         localMem[i] = 0;
      }
      sycl::group_barrier(group);

      for (auto i = globalId; i < bulkSize; i += stride) {
         // Fill local histogram
         auto bin = GetBin<Dim>(i, binEdges, binEdgesIdx, nBinsAxis, xMin, xMax, coords, bulkSize, bins);

         if (bin >= 0) {
            AddBinContent<sycl::access::address_space::local_space>(localMem, bin, weights[i]);
         }
      }
      sycl::group_barrier(group);

      // Merge results in global histogram
      for (auto i = localId; i < nBins; i += groupSize) {
         AddBinContent<sycl::access::address_space::global_space>(histogram, i, localMem[i]);
      }
   }
};

///////////////////////////////////////////
/// Statistics calculation kernels

template <unsigned int Dim>
class ExcludeUOverflowKernel {
public:
   ExcludeUOverflowKernel(AccR<int> _bins, AccW<double> _weights, AccR<int> _nBinsAxis, std::size_t _bulkSize)
      : bins(_bins), weights(_weights), nBinsAxis(_nBinsAxis), bulkSize(_bulkSize)
   {
   }

   void operator()(sycl::id<1> id) const
   {
      if (bins[id] <= 0 || bins[id] >= nBinsAxis[id / bulkSize] - 1) {
         weights[id % bulkSize] = 0.;
      }
   }

private:
   AccR<int> bins;
   AccW<double> weights;
   AccR<int> nBinsAxis;
   std::size_t bulkSize;
};

class CombineStatsKernel {
public:
   CombineStatsKernel(double *_stats, double *_intermediate) : stats(_stats), intermediate(_intermediate) {}

   void operator()(sycl::id<1> id) const { stats[id] += intermediate[id]; }

private:
   double *stats;
   double *intermediate;
};

///////////////////////////////////////////
/// RHnSYCL

template <typename T, unsigned int Dim, unsigned int WGroupSize>
RHnSYCL<T, Dim, WGroupSize>::RHnSYCL(const std::size_t maxBulkSize, const std::size_t nBins,
                                     const std::array<int, Dim> &nBinsAxis, const std::array<double, Dim> &xLow,
                                     const std::array<double, Dim> &xHigh, const std::vector<double> &binEdges,
                                     const std::array<int, Dim> &binEdgesIdx)
   : queue(sycl::gpu_selector_v, SYCLHelpers::exception_handler),
     kStatsSmemSize((WGroupSize <= 32) ? 2 * WGroupSize * sizeof(double) : WGroupSize * sizeof(double))
{
   auto device = queue.get_device();

   fMaxBulkSize = maxBulkSize;
   fNBins = nBins;
   fEntries = 0;

   // Setup device memory for filling the histogram.
   fBWeights = sycl::buffer<double, 1>(sycl::range<1>(fMaxBulkSize));
   fBCoords = sycl::buffer<double, 1>(sycl::range<1>(Dim * fMaxBulkSize));
   fBBins = sycl::buffer<int, 1>(sycl::range<1>(Dim * fMaxBulkSize));

   // Setup device memory for histogram characteristics
   fBNBinsAxis = sycl::buffer<int, 1>(nBinsAxis.data(), sycl::range<1>(nBinsAxis.size()));
   queue.copy(nBinsAxis.data(), sycl::accessor{*fBNBinsAxis, sycl::write_only});
   fBMin = sycl::buffer<double, 1>(xLow.data(), sycl::range<1>(xLow.size()));
   queue.copy(xLow.data(), sycl::accessor{*fBMin, sycl::write_only});
   fBMax = sycl::buffer<double, 1>(xHigh.data(), sycl::range<1>(xHigh.size()));
   queue.copy(xHigh.data(), sycl::accessor{*fBMax, sycl::write_only});
   fBBinEdgesIdx = sycl::buffer<int, 1>(binEdgesIdx.data(), sycl::range<1>(binEdgesIdx.size()));
   queue.copy(binEdgesIdx.data(), sycl::accessor{*fBBinEdgesIdx, sycl::write_only});

   fDBinEdges = NULL;
   if (binEdges.size() > 0) {
      fDBinEdges = sycl::malloc_device<double>(binEdges.size(), queue);
      queue.memcpy(fDBinEdges, binEdges.data(), binEdges.size() * sizeof(double));
   }

   // Allocate and initialize device memory for the histogram and statistics.
   fBHistogram = sycl::buffer<T, 1>(sycl::range<1>(fNBins));
   SYCLHelpers::InitializeToZero(queue, *fBHistogram, fNBins);

   fDStats = sycl::malloc_device<double>(kNStats, queue);
   queue.memset(fDStats, 0, kNStats * sizeof(double)).wait();

   fDIntermediateStats = NULL;
#ifdef ROOT_RDF_ADAPTIVECPP
   fDIntermediateStats = sycl::malloc_device<double>(kNStats, queue);
   queue.memset(fDIntermediateStats, 0, kNStats * sizeof(double)).wait();
#endif

   queue.wait();

   // Determine the amount of shared memory required for HistogramKernel, and the maximum available.
   fHistoSmemSize = fNBins * sizeof(T);
   auto has_local_mem = device.is_gpu() || (device.template get_info<sycl::info::device::local_mem_type>() !=
                                            sycl::info::local_mem_type::none);
   fMaxSmemSize = has_local_mem ? device.template get_info<sycl::info::device::local_mem_size>() : 0;

   if (getenv("DBG")) {
      std::cout << "Running SYCLHist on " << device.template get_info<sycl::info::device::name>() << "\n";
      printf("USM support: %s\n", device.has(sycl::aspect::usm_device_allocations) ? "yes" : "no");
      printf("Maximum shared memory size: %zu\n", fMaxSmemSize);
   }
}

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::Fill(const RVecD &coords)
{
   RVecD weights(coords.size() / Dim, 1);
   Fill(coords, weights);
}

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::Fill(const RVecD &coords, const RVecD &weights)
{
   auto bulkSize = weights.size();

   // Add the coordinates and weight to the buffers
   {
      sycl::host_accessor coordsAcc{*fBCoords, sycl::write_only, sycl::no_init};
      sycl::host_accessor weightsAcc{*fBWeights, sycl::write_only, sycl::no_init};
      std::copy(coords.begin(), coords.end(), coordsAcc.get_pointer());
      std::copy(weights.begin(), weights.end(), weightsAcc.get_pointer());
   }

   fEntries += bulkSize;

   // The histogram kernels execute asynchronously.
   ExecuteSYCLHisto(bulkSize);
}

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::GetStats(std::size_t size)
{
   // Set weights of over/underflow bins to zero. Done in separate kernel in case we want to add the option to add
   // under/overflow bins to the statistics.
   queue.submit([&](sycl::handler &cgh) {
      sycl::accessor binsAcc{*fBBins, cgh, sycl::range<1>(size * Dim), sycl::read_only};
      sycl::accessor weightsAcc{*fBWeights, cgh, sycl::range<1>(size), sycl::write_only};
      sycl::accessor nBinsAxisAcc{*fBNBinsAxis, cgh, sycl::read_only};
      cgh.parallel_for(sycl::range<1>(size * Dim),
                       ExcludeUOverflowKernel<Dim>(binsAcc, weightsAcc, nBinsAxisAcc, size));
   });

   std::vector<sycl::event> statsReductions;
   std::size_t reductionRange = ceil(size / 8.);
   auto resultPtr = fDStats;
#ifdef ROOT_RDF_ADAPTIVECPP
   resultPtr = fDIntermediateStats;
#endif

   statsReductions.push_back(queue.submit([&](sycl::handler &cgh) {
      sycl::accessor weightsAcc{*fBWeights, cgh, sycl::range<1>(size), sycl::read_only};
      auto GetSumW = sycl::reduction(&resultPtr[0], sycl::plus<double>());
      auto GetSumW2 = sycl::reduction(&resultPtr[1], sycl::plus<double>());

      cgh.parallel_for(sycl::range<1>(reductionRange), GetSumW, GetSumW2, [=](sycl::id<1> id, auto &sumw, auto &sumw2) {
         for (unsigned int gid = id; gid < size; gid += reductionRange) {
            sumw += weightsAcc[gid];
            sumw2 += weightsAcc[gid] * weightsAcc[gid];
         }
      });
   }));

   auto offset = 2;
   for (auto d = 0U; d < Dim; d++) {
      statsReductions.push_back(queue.submit([&](sycl::handler &cgh) {
         sycl::accessor coordsAcc{*fBCoords, cgh, sycl::range<1>(size * Dim), sycl::read_only};
         sycl::accessor weightsAcc{*fBWeights, cgh, sycl::range<1>(size), sycl::read_only};

         // Multiply weight with coordinate of current axis. E.g., for Dim = 2 this computes Tsumwx and Tsumwy
         auto GetSumWAxis = sycl::reduction(&resultPtr[offset++], sycl::plus<double>());
         auto GetSumWAxis2 = sycl::reduction(&resultPtr[offset++], sycl::plus<double>());

         cgh.parallel_for(sycl::range<1>(reductionRange), GetSumWAxis, GetSumWAxis2,
                          [=](sycl::id<1> id, auto &sumwaxis, auto &sumwaxis2) {
                             for (unsigned int gid = id; gid < size; gid += reductionRange) {
                                sumwaxis += weightsAcc[gid] * coordsAcc[d * size + gid];
                                sumwaxis2 += weightsAcc[gid] * coordsAcc[d * size + gid] * coordsAcc[d * size + gid];
                             }
                          });
      }));

      for (auto prev_d = 0U; prev_d < d; prev_d++) {
         statsReductions.push_back(queue.submit([&](sycl::handler &cgh) {
            sycl::accessor coordsAcc{*fBCoords, cgh, sycl::range<1>(size * Dim), sycl::read_only};
            sycl::accessor weightsAcc{*fBWeights, cgh, sycl::range<1>(size), sycl::read_only};

            // Multiplies coordinate of current axis with the "previous" axis. E.g., for Dim = 2 this computes
            // Tsumwxy
            auto GetSumWAxisAxis = sycl::reduction(&resultPtr[offset++], sycl::plus<double>());

            cgh.parallel_for(sycl::range<1>(reductionRange), GetSumWAxisAxis, [=](sycl::id<1> id, auto &sumwaxisaxis) {
               for (unsigned int gid = id; gid < size; gid += reductionRange) {
                  sumwaxisaxis += weightsAcc[gid] * coordsAcc[d * size + gid] * coordsAcc[prev_d * size + gid];
               }
            });
         }));
      }
   }

#ifdef ROOT_RDF_ADAPTIVECPP
   // The AdaptiveCpp reduction interface overwrites the output array instead of combining the original value,
   // so we have to add the values to the previously reduced values.
   auto combineEvent = queue.submit([&](sycl::handler &cgh) {
      // Explicit dependency required because dependencies are only defined implicitly when creating accessors,
      // but we don't create an accessor on device pointers.
      cgh.depends_on(statsReductions);
      cgh.parallel_for(sycl::range<1>(kNStats), CombineStatsKernel(fDStats, fDIntermediateStats));
   });
#endif
}

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::ExecuteSYCLHisto(std::size_t size)
{
   if (fHistoSmemSize > fMaxSmemSize) {
      queue.submit([&](sycl::handler &cgh) {
         // Get handles to SYCL buffers.
         sycl::accessor histogramAcc{*fBHistogram, cgh, sycl::read_write};
         sycl::accessor binEdgesIdxAcc{*fBBinEdgesIdx, cgh, sycl::read_only};
         sycl::accessor nBinsAxisAcc{*fBNBinsAxis, cgh, sycl::read_only};
         sycl::accessor minAcc{*fBMin, cgh, sycl::read_only};
         sycl::accessor maxAcc{*fBMax, cgh, sycl::read_only};
         sycl::accessor coordsAcc{*fBCoords, cgh, sycl::range<1>(size * Dim), sycl::read_only};
         sycl::accessor weightsAcc{*fBWeights, cgh, sycl::range<1>(size), sycl::read_only};
         sycl::accessor binsAcc{*fBBins, cgh, sycl::range<1>(size * Dim), sycl::write_only, sycl::no_init};

         // Partitions the vector pairs over available threads and computes the invariant masses.
         cgh.parallel_for(sycl::range<1>(size),
                          HistogramGlobal<T, Dim>(histogramAcc, fDBinEdges, binEdgesIdxAcc, nBinsAxisAcc, minAcc,
                                                  maxAcc, coordsAcc, weightsAcc, binsAcc, size));
      });
   } else {
      queue.submit([&](sycl::handler &cgh) {
         // Similar to CUDA shared memory.
         sycl::local_accessor<T, 1> localMem(sycl::range<1>(fNBins), cgh);

         // Get handles to SYCL buffers.
         sycl::accessor histogramAcc{*fBHistogram, cgh, sycl::read_write};
         sycl::accessor binEdgesIdxAcc{*fBBinEdgesIdx, cgh, sycl::read_only};
         sycl::accessor nBinsAxisAcc{*fBNBinsAxis, cgh, sycl::read_only};
         sycl::accessor minAcc{*fBMin, cgh, sycl::read_only};
         sycl::accessor maxAcc{*fBMax, cgh, sycl::read_only};
         sycl::accessor coordsAcc{*fBCoords, cgh, sycl::range<1>(size * Dim), sycl::read_only};
         sycl::accessor weightsAcc{*fBWeights, cgh, sycl::range<1>(size), sycl::read_only};
         sycl::accessor binsAcc{*fBBins, cgh, sycl::range<1>(size * Dim), sycl::write_only, sycl::no_init};

         // Global range must be a multiple of local range (WGroupSize) that is equal or larger than local range.
         auto execution_range = sycl::nd_range<1>{sycl::range<1>{((size + WGroupSize - 1) / WGroupSize) * WGroupSize},
                                                  sycl::range<1>{WGroupSize}};

         cgh.parallel_for(execution_range,
                          HistogramLocal<T, Dim>(localMem, histogramAcc, fDBinEdges, binEdgesIdxAcc, nBinsAxisAcc,
                                                 minAcc, maxAcc, coordsAcc, weightsAcc, binsAcc, fNBins, size));
      });
   } // end of scope, ensures data copied back to host

   GetStats(size);
}

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::RetrieveResults(T *histResult, double *statsResult)
{
   queue.wait();
   queue.copy(sycl::accessor{*fBHistogram, sycl::read_only}, histResult);
   queue.memcpy(statsResult, fDStats, kNStats * sizeof(double));
   queue.wait_and_throw();
}

#include "RHnSYCL-impl.cxx"

} // namespace Experimental
} // namespace ROOT
