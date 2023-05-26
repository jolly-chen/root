#include "RHnSYCL.h"
#include <CL/sycl.hpp>
#include <SYCLHelpers.h>
#include <iostream>
#include <array>
#include "TMath.h"

namespace sycl = cl::sycl;

namespace ROOT {
namespace Experimental {

class histogram_local;

inline int FindFixBin(double x, const double *binEdges, int nBins, double xMin, double xMax)
{
   int bin;

   // OPTIMIZATION: can this be done with less branching?
   if (x < xMin) { // underflow
      bin = 0;
   } else if (!(x < xMax)) { // overflow  (note the way to catch NaN)
      bin = nBins + 1;
   } else {
      if (binEdges == NULL) // fix bins
         bin = 1 + int(nBins * (x - xMin) / (xMax - xMin));
      else // variable bin sizes
         bin = 1 + SYCLHelpers::BinarySearch(nBins + 1, binEdges, x);
   }

   return bin;
}

template <unsigned int Dim>
inline int GetBin(int i, AxisDescriptor *axes, double *coords, int *bins)
{
   auto *x = &coords[i * Dim];

   auto bin = 0;
   for (int d = Dim - 1; d >= 0; d--) {
      auto binD = FindFixBin(x[d], axes[d].kBinEdges, axes[d].fNbins - 2, axes[d].fMin, axes[d].fMax);
      bins[i * Dim + d] = binD;

      if (binD < 0) {
         return -1;
      }

      bin = bin * axes[d].fNbins + binD;
   }

   return bin;
}

// template <typename AccHist>
// class HistogramGlobal {
// public:
//    HistogramGlobal(AccHist _hAcc, sycl::accessor<double, 1, sycl::access::mode::read> _cAcc,
//                    sycl::accessor<double, 1, sycl::access::mode::read> _wAcc)
//       : hAcc(_hAcc), cAcc(_cAcc), wAcc(_wAcc)
//    {
//    }
//    void operator()(sycl::item<1> item)
//    {
//       size_t id = item.get_linear_id();
//       // auto bin = GetBin(id, NULL, cAcc, NULL);
//    }

// protected:
//    AccHist hAcc;
//    sycl::accessor<double, 1, sycl::access::mode::read> cAcc, wAcc;
// };

// // Deriving from HistogramGlobal gives [Computecpp:CC0012]: class ROOT::Experimental::HistogramLocal cannot be a
// // parameter to a SYCL kernel, because it is a non standard-layout type
// template <typename AccHist>
// class HistogramLocal : public HistogramGlobal<AccHist> {
// public:
//    HistogramLocal(AccHist _hAcc, sycl::accessor<double, 1, sycl::access::mode::read> _cAcc,
//                   sycl::accessor<double, 1, sycl::access::mode::read> _wAcc)
//       : HistogramGlobal<AccHist>(_hAcc, _cAcc, _wAcc)
//    {
//    }

//    void operator()(sycl::item<1> item)
//    {
//       size_t id = item.get_linear_id();
//       // auto bin = GetBin(id, NULL, cAcc, NULL);
//    }
// };

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::AllocateBuffers()
{
   fHistogram = sycl::buffer<T, 1>(sycl::range<1>(fNbins));
   fWeights = sycl::buffer<double, 1>(sycl::range<1>(fBufferSize));
   fCoords = sycl::buffer<double, 1>(sycl::range<1>(Dim * fBufferSize));
   fBins = sycl::buffer<int, 1>(sycl::range<1>(fNbins));
   fAxes = sycl::buffer<AxisDescriptor, 1>(sycl::range<1>(Dim));
   fStats = sycl::buffer<double, 1>(sycl::range<1>(kNStats));
   fBinEdges = sycl::buffer<double, 1>(sycl::range<1>(Dim * fNbins));
}

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
   queue = sycl::queue(sycl::cpu_selector(), SYCLHelpers::exception_handler);
   auto device = queue.get_device();
   std::cout << "Running SYCLHist on " << device.template get_info<sycl::info::device::name>() << "\n";

   fBufferSize = 10000;
   fNbins = 1;
   fEntries = 0;

   AllocateBuffers();

   // There is no memset for buffers so we use kernels to initialize to zero.
   try {
      queue.submit([&](sycl::handler &cgh) {
         sycl::stream out(1024, 256, cgh);

         auto hAcc = fHistogram.template get_access<sycl::access::mode::discard_write>(cgh);
         auto sAcc = fStats.get_access<sycl::access::mode::discard_write>(cgh);

         cgh.single_task(SYCLHelpers::PrintArray(out, sAcc, "sAcc"));
         // cgh.single_task(SYCLHelpers::PrintArray(out, hAcc, "hAcc"));
         // cgh.parallel_for(sycl::range<1>(fNbins), SYCLHelpers::InitializeZero(hAcc));
         // cgh.parallel_for(sycl::range<1>(kNStats), SYCLHelpers::InitializeZero(sAcc));
      });
   } catch (sycl::exception const &e) {
      std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
   }

   // auto axesAcc = fAxes.template get_access<sycl::access::mode::discard_write>();         // Host access
   // auto binEdgesAcc = fBinEdges.template get_access<sycl::access::mode::discard_write>(); // Host access

   // // Initialize axis descriptors.
   // for (unsigned int i = 0; i < Dim; i++) {
   //    AxisDescriptor axis;
   //    axis.fNbins = ncells[i];
   //    axis.fMin = xlow[i];
   //    axis.fMax = xhigh[i];
   //    axis.kBinEdges = NULL;
   //    // if (binEdges != NULL)

   //    axesAcc[i] = axis;
   //    fNbins *= ncells[i];
   // }

   // fHistoSmemSize = fNbins * sizeof(T);
   // auto has_local_mem = device.is_host() || (device.template get_info<sycl::info::device::local_mem_type>() !=
   //                                           sycl::info::local_mem_type::none);
   // fMaxSmemSize = has_local_mem ? device.template get_info<sycl::info::device::local_mem_size>() : 0;
}

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::Fill(const std::array<double, Dim> &coords, double w)
{
   // Host accessors
   // auto coordsAcc = fCoords.get_access<sycl::access::mode::read_write>();
   // auto weightsAcc = fWeights.get_access<sycl::access::mode::read_write>();

   // fEntries++;
   // auto bufferIdx = fEntries % fBufferSize;

   // for (unsigned int i = 0; i < Dim; i++) {
   //    coordsAcc[bufferIdx * Dim + i] = coords[i];
   // }
   // weightsAcc[bufferIdx] = w;

   // // Only execute when a certain number of values are buffered to increase the GPU workload and decrease the
   // // frequency of kernel launches.
   // if (bufferIdx == fBufferSize) {
   //    ExecuteSYCLHisto();
   // }
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
   // unsigned int size = fmin(fBufferSize, fEntries % fBufferSize);
   // int numWGroups = round(size / (float)WGroupSize);

   // if (fHistoSmemSize > fMaxSmemSize) {
      // queue.submit([&](sycl::handler &cgh) {
      //    // Get handles to SYCL buffers.
      //    auto wAcc = fWeights.get_access<sycl::access::mode::read>(cgh);
      //    auto cAcc = fCoords.get_access<sycl::access::mode::read>(cgh);
      //    auto hAcc = fHistogram.template get_access<sycl::access::mode::read_write>(cgh);

      //    // Partitions the vector pairs over available threads and computes the invariant masses.
      //    cgh.parallel_for(sycl::range<1>(size), HistogramGlobal(hAcc, wAcc, cAcc));
      // });
   // } else {
      // queue.submit([&](sycl::handler &cgh) {
      //    // Similar to CUDA shared memory.
      //    // auto localMemRange = sycl::range<1>(WGroupSize);
      //    // sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> LocalMem(localMemRange,
      //    cgh);

      //    // Get handles to SYCL buffers.
      //    auto wAcc = fWeights.get_access<sycl::access::mode::read>(cgh);
      //    auto cAcc = fCoords.get_access<sycl::access::mode::read>(cgh);
      //    auto hAcc = fHistogram.template get_access<sycl::access::mode::read_write>(cgh);

      //    cgh.parallel_for(sycl::range<1>(size), HistogramGlobal(hAcc, cAcc, wAcc));
      // });
   // } // end of scope, ensures data copied back to host
}

template <typename T, unsigned int Dim, unsigned int WGroupSize>
void RHnSYCL<T, Dim, WGroupSize>::RetrieveResults(T *histResult, double *statsResult)
{
   // Fill the histogram with remaining values in the buffer.
   // auto weightsAcc = fWeights.get_access<sycl::access::mode::read>(); // Host access
   // if (weightsAcc.get_size() > 0) {
   //    try {
   //       ExecuteSYCLHisto();
   //       queue.wait_and_throw();
   //    } catch (sycl::exception const &e) {
   //       std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
   //    }
   // }

   // TODO: Free device pointers?
   // queue.copy(fHistogram, histResult);
   // queue.copy(fStats, statsResult);
}

#include "RHnSYCL-impl.cxx"

} // namespace Experimental
} // namespace ROOT
