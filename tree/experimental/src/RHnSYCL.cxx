#include "RHnGPU.h"
#include "RHnSYCL.h"
#include <CL/sycl.hpp>
#include <SYCLHelpers.h>
#include <iostream>
#include <array>
#include <vector>
#include "TMath.h"


namespace sycl = cl::sycl;

namespace ROOT {
namespace Experimental {

#include "RHnSYCL-impl.cxx"

class histogram_sycl;

template <typename T, unsigned int Dim, unsigned int BlockSize>
RHnSYCL<T, Dim, BlockSize>::RHnSYCL(std::array<int, Dim> ncells, std::array<double, Dim> xlow,
                                    std::array<double, Dim> xhigh, const double **binEdges)
   : kNStats([]() {
        // Sum of weights (squared) + sum of weight * bin (squared) per axis + sum of weight * binAx1 * binAx2 for
        // all axis combinations
        return Dim > 1 ? 2 + 2 * Dim + TMath::Binomial(Dim, 2) : 2 + 2 * Dim;
     }()),
     kStatsSmemSize((BlockSize <= 32) ? 2 * BlockSize * sizeof(double) : BlockSize * sizeof(double)),
     queue(sycl::default_selector{}, SYCLHelpers::exception_handler)
{
   fBufferSize = 10000;
   fNbins = 1;
   fEntries = 0;

   AllocateBuffers();
   auto axesAcc = fAxes.template get_access<sycl::access::mode::discard_write>();

   // Initialize axis descriptors.
   for (auto i = 0; i < Dim; i++) {
      RAxis axis;
      axis.fNbins = ncells[i];
      axis.fMin = xlow[i];
      axis.fMax = xhigh[i];
      if (binEdges != NULL)
         axis.kBinEdges = binEdges[i];
      else
         axis.kBinEdges = NULL;

      axesAcc[i] = axis;
      fNbins *= ncells[i];
   }

   auto device = queue.get_device();
   std::cout << "Running SYCLHist on " << device.template get_info<sycl::info::device::name>() << "\n";

   fHistoSmemSize = fNbins * sizeof(T);
   auto has_local_mem = device.is_host() || (device.template get_info<sycl::info::device::local_mem_type>() !=
                                             sycl::info::local_mem_type::none);
   fMaxSmemSize = has_local_mem ? device.template get_info<sycl::info::device::local_mem_size>() : 0;
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnSYCL<T, Dim, BlockSize>::AllocateBuffers()
{
   fHistogram = sycl::buffer<T, 1>(sycl::range<1>(fNbins));
   fWeights = sycl::buffer<double, 1>(sycl::range<1>(fBufferSize));
   fCoords = sycl::buffer<double, 1>(sycl::range<1>(Dim * fBufferSize));
   fBins = sycl::buffer<int, 1>(sycl::range<1>(fNbins));
   fAxes = sycl::buffer<RAxis, 1>(sycl::range<1>(Dim));
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnSYCL<T, Dim, BlockSize>::Fill(const std::array<double, Dim> &coords, double w)
{
   auto coordsAcc = fCoords.get_access<sycl::access::mode::read_write>();
   auto weightsAcc = fWeights.get_access<sycl::access::mode::read_write>();

   // coordsAcc.insert(coordsAcc.end(), coords.begin(), coords.end());
   // weightsAcc.push_back(w);

   // Only execute when a certain number of values are buffered to increase the GPU workload and decrease the
   // frequency of kernel launches.
   if (weightsAcc.get_size() == fBufferSize) {
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

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnSYCL<T, Dim, BlockSize>::GetStats(unsigned int size)
{
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnSYCL<T, Dim, BlockSize>::ExecuteSYCLHisto()
{
   if (fHistoSmemSize > fMaxSmemSize) {
      throw "Device doesn't have enough local memory!";
   } else {
      // Start of scope, ensures data copied back to host
      // Create device buffers. The memory is managed by SYCL so we should NOT access these buffers directly.

      queue.submit([&](sycl::handler &cgh) {
         // Get handles to SYCL buffers.
         auto wAcc = fWeights.get_access<sycl::access::mode::read>(cgh);
         auto cAcc = fCoords.get_access<sycl::access::mode::read>(cgh);
         auto hAcc = fHistogram.template get_access<sycl::access::mode::read_write>(cgh);
      });
   } // end of scope, ensures data copied back to host
}

template <typename T, unsigned int Dim, unsigned int BlockSize>
void RHnSYCL<T, Dim, BlockSize>::RetrieveResults(T *histResult, double *statsResult)
{
   // Fill the histogram with remaining values in the buffer.
   auto weightsAcc = fWeights.get_access<sycl::access::mode::read>();
   if (weightsAcc.get_size() > 0) {
      ExecuteSYCLHisto();
   }

   try {
      queue.wait_and_throw();
   } catch (sycl::exception const &e) {
      std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
   }

   // TODO: Free device pointers?
}

} // namespace Experimental
} // namespace ROOT
