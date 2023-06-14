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

// inline int FindFixBin(double x, int nBins, double xMin, double xMax)
// {
//    int bin;

//    // OPTIMIZATION: can this be done with less branching?
//    if (x < xMin) { // underflow
//       bin = 0;
//    } else if (!(x < xMax)) { // overflow  (note the way to catch NaN)
//       bin = nBins + 1;
//    } else {
//       bin = 1 + int(nBins * (x - xMin) / (xMax - xMin));
//    }

//    return bin;
// }

// inline int FindFixBin(double x, int nBins, double xMin, double xMax, const double *binEdges)
// {
//    int bin;

//    if (x < xMin) { // underflow
//       bin = 0;
//    } else if (!(x < xMax)) { // overflow  (note the way to catch NaN)
//       bin = nBins + 1;
//    } else { // variable bin sizes
//       bin = 1 + SYCLHelpers::BinarySearch(nBins + 1, binEdges, x);
//    }

//    return bin;
// }

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

template <typename T, unsigned int Dim>
class HistogramGlobal {
   using AccHistT = sycl::accessor<T, 1, mode::read_write>;
   // using AccHistT = T *;

public:
   HistogramGlobal(AccHistT _histogramAcc, AccAxesR _axesAcc, AccDoubleR _coordsAcc, AccDoubleR _weightsAcc,
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
      // printf("id: %d bin: %d weight:%f\n", id, bin, weightsAcc[id]);

      // if (id == 0) {
      //    printf("weights:\n");
      //    for (int i = 0; i < 5; i++) {
      //       printf("%f ", weightsAcc[i]);
      //    }
      //    printf("\n");

      //    printf("coords:\n");
      //    for (int i = 0; i < 5; i++) {
      //       printf("%f ", coordsAcc[i]);
      //    }
      //    printf("\n");

      //    printf("axes:\n");
      //    for (unsigned int i = 0; i < Dim; i++) {
      //       printf("\tdim: %d nbins: %d min: %f max: %f idx: %d binedges:", i, axesAcc[i].fNbins, axesAcc[i].fMin,
      //              axesAcc[i].fMax, axesAcc[i].binEdgesIdx);
      //       if (binEdges == NULL) {
      //          printf("NULL\n");
      //       } else {
      //          for (int j = 0; j < axesAcc[i].fNbins - 1; j++) {
      //             printf("%f ", binEdges[axesAcc[i].binEdgesIdx + j]);
      //          }
      //          printf("\n");
      //       }
      //    }
      //    printf("\n");
      // }

      auto hAtomic = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>(histogramAcc[bin]);
      hAtomic.fetch_add((T)weightsAcc[id]);

      // if (id == 0) {
      //    for (int i = 0; i < 4; i++) {
      //       printf("%f ", histogramAcc[i]);
      //    }
      // }
   }

protected:
   AccHistT histogramAcc;
   AccAxesR axesAcc;
   AccDoubleR coordsAcc, weightsAcc;
   AccBinsW binsAcc;
   const double *binEdges;
};

// template <unsigned int Dim, class AccHistT>
// class HistogramLocal : public HistogramGlobal<Dim, AccHistT> {
// public:
//    HistogramLocal(AccHistT _histogramAcc, AccAxesR _axesAcc, AccDoubleR _binEdgesAcc, AccDoubleR _coordsAcc,
//                   AccDoubleR _weightsAcc, AccBinsW _binsAcc)
//       : HistogramGlobal<Dim, AccHistT>(_histogramAcc, _axesAcc, _binEdgesAcc, _coordsAcc, _weightsAcc, _binsAcc)
//    {
//    }

//    void operator()(sycl::item<1> item) const
//    {
//       // size_t id = item.get_linear_id();
//       // auto bin = GetBin<Dim>(id, axesAcc, coordsAcc, binsAcc);
//    }
// };

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
   queue = sycl::queue(sycl::gpu_selector{}, SYCLHelpers::exception_handler);
   // queue = sycl::queue(sycl::gpu_selector{}, SYCLHelpers::exception_handler);
   auto device = queue.get_device();
   std::cout << "Running SYCLHist on " << device.template get_info<sycl::info::device::name>() << "\n";

   fBufferSize = 10000;
   fNbins = 1;
   fEntries = 0;

   // Allocate buffers
   fBWeights = new sycl::buffer<double, 1>(sycl::range<1>(fBufferSize));
   fBCoords = new sycl::buffer<double, 1>(sycl::range<1>(Dim * fBufferSize));
   fBBins = new sycl::buffer<int, 1>(sycl::range<1>(fBufferSize * Dim));
   fBAxes = new sycl::buffer<AxisDescriptor, 1>(sycl::range<1>(Dim));
   std::vector<double> binEdgesFlat;
   int numBinEdges = 0;

   // Initialize axis descriptors.
   {
      sycl::host_accessor axesAcc{*fBAxes, sycl::write_only, sycl::no_init};
      for (unsigned int i = 0; i < Dim; i++) {
         AxisDescriptor axis;
         axis.fNbins = ncells[i];
         axis.fMin = xlow[i];
         axis.fMax = xhigh[i];
         axis.kBinEdges = NULL;

         if (binEdges[i] != NULL) {
            binEdgesFlat.insert(binEdgesFlat.end(), binEdges[i], binEdges[i] + (ncells[i] - 1));
            axis.binEdgesIdx = numBinEdges;
            numBinEdges += ncells[i] - 1;
         } else {
            axis.binEdgesIdx = -1;
         }

         axesAcc[i] = axis;
         fNbins *= ncells[i];
      }
   }

   // Allocate and initialize buffers for the histogram and statistics.
   fBHistogram = new sycl::buffer<T, 1>(sycl::range<1>(fNbins));
   SYCLHelpers::InitializeZero(queue, *fBHistogram, fNbins);
   fBStats = new sycl::buffer<double, 1>(sycl::range<1>(kNStats));
   SYCLHelpers::InitializeZero(queue, *fBStats, fNbins);

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
      sycl::host_accessor coordsAcc{*fBCoords, sycl::write_only, sycl::no_init};
      sycl::host_accessor weightsAcc{*fBWeights, sycl::write_only, sycl::no_init};
      for (unsigned int i = 0; i < Dim; i++) {
         coordsAcc[bufferIdx * Dim + i] = coords[i];
      }
      weightsAcc[bufferIdx] = w;
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

   // if (fHistoSmemSize > fMaxSmemSize) {
   queue.submit([&](sycl::handler &cgh) {
      // Get handles to SYCL buffers.
      sycl::accessor histogramAcc{*fBHistogram, cgh, sycl::read_write};
      sycl::accessor axesAcc{*fBAxes, cgh, sycl::read_only};
      sycl::accessor weightsAcc{*fBWeights, cgh, sycl::read_only};
      sycl::accessor coordsAcc{*fBCoords, cgh, sycl::read_only};
      sycl::accessor binsAcc{*fBBins, cgh, sycl::write_only, sycl::no_init};

      // Partitions the vector pairs over available threads and computes the invariant masses.
      cgh.parallel_for(sycl::range<1>(size),
                       HistogramGlobal<T, Dim>(histogramAcc, axesAcc, coordsAcc, weightsAcc, binsAcc, fDBinEdges));
   });
   // } else {
   // queue.submit([&](sycl::handler &cgh) {
   //    // Similar to CUDA shared memory.
   //    // auto localMemRange = sycl::range<1>(WGroupSize);
   //    // sycl::accessor<T, 1, mode::read_write, sycl::access::target::local> LocalMem(localMemRange,
   //    cgh);

   //    // Get handles to SYCL buffers.
   //    auto weightsAcc = fBWeights->get_access<mode::read>(cgh);
   //    auto coordsAcc = fBCoords->get_access<mode::read>(cgh);
   //    auto hAcc = fBHistogram.template get_access<mode::read_write>(cgh);

   //    cgh.parallel_for(sycl::range<1>(size), HistogramGlobal(hAcc, coordsAcc, weightsAcc));
   // });
   // } // end of scope, ensures data copied back to host

   // SYCLHelpers::PrintArray(queue, fBHistogram, fNbins);
}

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::RetrieveResults(T *histResult, double *statsResult)
{
   // Fill the histogram with remaining values in the buffer.
   if (fEntries % fBufferSize != 0) {
      ExecuteSYCLHisto();
   }

   try {
      queue.copy(sycl::accessor{*fBHistogram, sycl::read_only}, histResult);
      queue.wait_and_throw();
   } catch (sycl::exception const &e) {
      std::cout << "Caught synchronous SYCL exception"
                << " :" << __LINE__ << ":\n"
                << e.what() << std::endl;
   }
}

#include "RHnSYCL-impl.cxx"

} // namespace Experimental
} // namespace ROOT
