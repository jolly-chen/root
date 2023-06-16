#include "RHnSYCL.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <array>
#include "TMath.h"
#include "SYCLHelpers.h"

namespace ROOT {
namespace Experimental {

class histogram_local;

using mode = sycl::access::mode;

using AccDoubleR = sycl::accessor<double, 1, mode::read>;
using AccBinsW = sycl::accessor<int, 1, mode::write>;
using AccAxesR = sycl::accessor<AxisDescriptor, 1, mode::read>;

template <class T>
using AccHistRW = sycl::accessor<T, 1, mode::read_write>;

template <class T>
using AccLocalMem = sycl::accessor<T, 1, mode::read_write, sycl::access::target::local>;

inline int FindFixBin(double x, const double *binEdges, int binEdgesIdx, int nBins, double xMin, double xMax)
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
inline int GetBin(int i, AxisDescriptor *axes, double *coords, int *bins, const double *binEdges)
{
   auto *x = &coords[i * Dim];

   auto bin = 0;
   for (int d = Dim - 1; d >= 0; d--) {
      auto binD = FindFixBin(x[d], binEdges, axes[d].binEdgesIdx, axes[d].fNbins - 2, axes[d].fMin, axes[d].fMax);
      bins[i * Dim + d] = binD;

      if (binD < 0) {
         return -1;
      }

      bin = bin * axes[d].fNbins + binD;
   }

   return bin;
}

///////////////////////////////////////////
/// Methods for incrementing a bin.

template <sycl::access::address_space Space>
inline void AddBinContent(AccHistRW<double> histogram, int bin, double weight)
{
   auto atomic =
      sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
   atomic.fetch_add(weight);
}

template <sycl::access::address_space Space>
inline void AddBinContent(AccHistRW<float> histogram, int bin, double weight)
{
   auto atomic =
      sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
   atomic.fetch_add((float)weight);
}

template <sycl::access::address_space Space>
inline void AddBinContent(AccHistRW<short> histogram, int bin, double weight)
{
   // There is no fetch_add for short so we need to operate on integers... (Assumes little endian)
   short *addr = &histogram.get_pointer()[bin];
   int *addrInt = (int *)((char *)addr - ((size_t)addr & 2));
   int assumed, newVal, overwrite;
   bool success = false;

   do {
      assumed = histogram[bin];

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

      auto atomic = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(*addrInt);
      success = atomic.compare_exchange_strong(assumed, overwrite);
   } while (!success);
}

template <sycl::access::address_space Space>
inline void AddBinContent(AccHistRW<int> histogram, int bin, double weight)
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

template <typename T, unsigned int Dim>
class HistogramGlobal {
public:
   HistogramGlobal(AccHistRW<T> _histogramAcc, AccAxesR _axesAcc, AccDoubleR _coordsAcc, AccDoubleR _weightsAcc,
                   AccBinsW _binsAcc, const double *_binEdges)
      : histogramAcc(_histogramAcc),
        axesAcc(_axesAcc),
        coordsAcc(_coordsAcc),
        weightsAcc(_weightsAcc),
        binsAcc(_binsAcc),
        binEdges(_binEdges)
   {
   }

   void operator()(sycl::item<1> item) const
   {
      size_t id = item.get_linear_id();
      auto bin = GetBin<Dim>(id, axesAcc.get_pointer(), coordsAcc.get_pointer(), binsAcc.get_pointer(), binEdges);

      if (bin >= 0) {
         AddBinContent<sycl::access::address_space::global_space>(histogramAcc, bin, weightsAcc[id]);
      }
   }

protected:
   AccHistRW<T> histogramAcc;
   AccAxesR axesAcc;
   AccDoubleR coordsAcc, weightsAcc;
   AccBinsW binsAcc;
   const double *binEdges;
};

template <typename T, unsigned int Dim>
class HistogramLocal : public HistogramGlobal<T, Dim> {
public:
   HistogramLocal(AccLocalMem<T> _localMem, AccHistRW<T> _histogramAcc, AccAxesR _axesAcc, AccDoubleR _coordsAcc,
                  AccDoubleR _weightsAcc, AccBinsW _binsAcc, const double *_binEdges)
      : HistogramGlobal<T, Dim>(_histogramAcc, _axesAcc, _coordsAcc, _weightsAcc, _binsAcc, _binEdges)
   {
      localMem = _localMem;
   }

   void operator()(sycl::nd_item<1> item) const
   {
      auto globalId = item.get_global_id(0);
      auto localId = item.get_local_id(0);
      auto group = item.get_group();
      auto groupSize = item.get_local_range(0);
      auto stride = groupSize * item.get_group_range(0);
      auto nBins = this->histogramAcc.size();
      auto nCoords = this->weightsAcc.size();

      // Initialize a local per-work-group histogram
      for (auto i = localId; i < nBins; i += groupSize) {
         localMem[i] = 0;
      }
      sycl::group_barrier(group);

      for (auto i = globalId; i < nCoords; i += stride) {
         // Fill local histogram
         auto bin = GetBin<Dim>(i, this->axesAcc.get_pointer(), this->coordsAcc.get_pointer(),
                                this->binsAcc.get_pointer(), this->binEdges);

         if (bin >= 0) {
            AddBinContent<sycl::access::address_space::local_space>(this->histogramAcc, bin, this->weightsAcc[i]);
         }
      }
      sycl::group_barrier(group);

      // Merge results in global histogram
      for (auto i = localId; i < nBins; i += groupSize) {
         AddBinContent<sycl::access::address_space::global_space>(this->histogramAcc, i, localMem[i]);
      }
   }

protected:
   AccLocalMem<T> localMem;
};

template <typename T, unsigned int Dim, unsigned int WGroupSize>
RHnSYCL<T, Dim, WGroupSize>::RHnSYCL(std::array<int, Dim> ncells, std::array<double, Dim> xlow,
                                     std::array<double, Dim> xhigh, const double **binEdges)
   : kNStats([]() {
        // Sum of weights (squared) + sum of weight * bin (squared) per axis + sum of weight * binAx1 * binAx2 for
        // all axis combinations
        return Dim > 1 ? 2 + 2 * Dim + TMath::Binomial(Dim, 2) : 2 + 2 * Dim;
     }()),
     kStatsSmemSize((WGroupSize <= 32) ? 2 * WGroupSize * sizeof(double) : WGroupSize * sizeof(double))
{
   // queue = sycl::queue(sycl::cpu_selector{}, SYCLHelpers::exception_handler);
   queue = sycl::queue(sycl::gpu_selector{}, SYCLHelpers::exception_handler);
   auto device = queue.get_device();
   std::cout << "Running SYCLHist on " << device.template get_info<sycl::info::device::name>() << "\n";

   fBufferSize = 10000;
   fNbins = 1;
   fEntries = 0;

   // Allocate buffers
   fBWeights = new sycl::buffer<double, 1>(sycl::range<1>(fBufferSize));
   fBCoords = new sycl::buffer<double, 1>(sycl::range<1>(Dim * fBufferSize));
   fBBins = new sycl::buffer<int, 1>(sycl::range<1>(Dim * fBufferSize));
   fBAxes = new sycl::buffer<AxisDescriptor, 1>(sycl::range<1>(Dim));
   std::vector<double> binEdgesFlat;
   int numBinEdges = 0;

   // Initialize axis descriptors.
   {
      sycl::host_accessor axesAcc{*fBAxes, sycl::write_only, sycl::no_init};
      for (unsigned int d = 0; d < Dim; d++) {
         AxisDescriptor axis;
         axis.fNbins = ncells[d];
         axis.fMin = xlow[d];
         axis.fMax = xhigh[d];
         axis.kBinEdges = NULL;

         if (binEdges != NULL && binEdges[d] != NULL) {
            binEdgesFlat.insert(binEdgesFlat.end(), binEdges[d], binEdges[d] + (ncells[d] - 1));
            axis.binEdgesIdx = numBinEdges;
            numBinEdges += ncells[d] - 1;
         } else {
            axis.binEdgesIdx = -1;
         }

         axesAcc[d] = axis;
         fNbins *= ncells[d];
      }
   }

   // Allocate and initialize buffers for the histogram and statistics.
   fBHistogram = new sycl::buffer<T, 1>(sycl::range<1>(fNbins));
   SYCLHelpers::InitializeToZero(queue, *fBHistogram, fNbins);
   fBStats = new sycl::buffer<double, 1>(sycl::range<1>(kNStats));
   SYCLHelpers::InitializeToZero(queue, *fBStats, kNStats);

   // Initialize BinEdges buffer.
   fDBinEdges = NULL;
   if (numBinEdges > 0) {
      fDBinEdges = (const double *)sycl::malloc_device(numBinEdges * sizeof(double), queue);
      queue.memcpy((void *)fDBinEdges, binEdgesFlat.data(), numBinEdges * sizeof(double));
      queue.wait();
   }

   // Determine the amount of shared memory required for HistogramKernel, and the maximum available.
   fHistoSmemSize = fNbins * sizeof(T);
   auto has_local_mem = device.is_host() || (device.template get_info<sycl::info::device::local_mem_type>() !=
                                             sycl::info::local_mem_type::none);
   fMaxSmemSize = has_local_mem ? device.template get_info<sycl::info::device::local_mem_size>() : 0;
}

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::Fill(const std::array<double, Dim> &coords, double w)
{
   auto bufferIdx = fEntries % fBufferSize;
   fEntries++;

   {
      sycl::host_accessor coordsAcc{*fBCoords, sycl::range<1>(Dim), sycl::id{bufferIdx * Dim}, sycl::write_only,
                                    sycl::no_init};
      sycl::host_accessor weightsAcc{*fBWeights, sycl::range<1>(1), sycl::id{bufferIdx}, sycl::write_only,
                                     sycl::no_init};
      for (unsigned int i = 0; i < Dim; i++) {
         coordsAcc[i] = coords[i];
      }
      weightsAcc[0] = w;
   }

   // Only execute when a certain number of values are buffered to increase the GPU workload and decrease the
   // frequency of kernel launches.
   if (bufferIdx == fBufferSize) {
      ExecuteSYCLHisto();
   }
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

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::GetStats(unsigned int size)
{
}

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::ExecuteSYCLHisto()
{
   unsigned int size = fmin(fBufferSize, fEntries % fBufferSize);

   if (fHistoSmemSize > fMaxSmemSize) {
      queue.submit([&](sycl::handler &cgh) {
         // Get handles to SYCL buffers.
         sycl::accessor histogramAcc{*fBHistogram, cgh, sycl::read_write};
         sycl::accessor axesAcc{*fBAxes, cgh, sycl::read_only};
         sycl::accessor weightsAcc{*fBWeights, cgh, sycl::range<1>(size), sycl::read_only};
         sycl::accessor coordsAcc{*fBCoords, cgh, sycl::range<1>(size * Dim), sycl::read_only};
         sycl::accessor binsAcc{*fBBins, cgh, sycl::range<1>(size * Dim), sycl::write_only, sycl::no_init};

         // Partitions the vector pairs over available threads and computes the invariant masses.
         cgh.parallel_for(sycl::range<1>(size),
                          HistogramGlobal<T, Dim>(histogramAcc, axesAcc, coordsAcc, weightsAcc, binsAcc, fDBinEdges));
      });
   } else {
      queue.submit([&](sycl::handler &cgh) {
         // Similar to CUDA shared memory.
         sycl::accessor<T, 1, mode::read_write, sycl::access::target::local> localMem(sycl::range<1>(fNbins), cgh);

         // Get handles to SYCL buffers.
         sycl::accessor histogramAcc{*fBHistogram, cgh, sycl::read_write};
         sycl::accessor axesAcc{*fBAxes, cgh, sycl::read_only};
         sycl::accessor weightsAcc{*fBWeights, cgh, sycl::range<1>(size), sycl::read_only};
         sycl::accessor coordsAcc{*fBCoords, cgh, sycl::range<1>(size * Dim), sycl::read_only};
         sycl::accessor binsAcc{*fBBins, cgh, sycl::range<1>(size * Dim), sycl::write_only, sycl::no_init};

         auto execution_range = sycl::nd_range<1>{sycl::range<1>{((size + WGroupSize - 1) / WGroupSize) * WGroupSize},
                                                  sycl::range<1>{WGroupSize}};

         cgh.parallel_for(execution_range, HistogramLocal<T, Dim>(localMem, histogramAcc, axesAcc, coordsAcc,
                                                                  weightsAcc, binsAcc, fDBinEdges));
      });
   } // end of scope, ensures data copied back to host
}

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::RetrieveResults(T *histResult, double *statsResult)
{
   // Fill the histogram with remaining values in the buffer.
   if (fEntries % fBufferSize != 0) {
      ExecuteSYCLHisto();
   }

   queue.copy(sycl::accessor{*fBHistogram, sycl::read_only}, histResult);
   queue.copy(sycl::accessor{*fBStats, sycl::read_only}, statsResult);
   queue.wait();
}

#include "RHnSYCL-impl.cxx"

} // namespace Experimental
} // namespace ROOT
