#ifndef RHnSYCL_H
#define RHnSYCL_H

#include <array>
#include <sycl/sycl.hpp>
#include "AxisDescriptor.h"

namespace ROOT {
namespace Experimental {

template <typename T, unsigned int Dim, unsigned int WGroupSize = 256>
class RHnSYCL {
   // clang-format off
   sycl::queue                      queue;

   sycl::buffer<T, 1>              *fBHistogram;         ///< Pointer to histogram buffer on the GPU.
   int                              fNbins;             ///< Total number of bins in the histogram WITH under/overflow

   const int                        kNStats;            ///< Number of statistics.
   sycl::buffer<AxisDescriptor, 1> *fBAxes;             ///< Vector of Dim axis descriptors
   const double                    *fDBinEdges;         ///< Binedges per axis for non-fixed bins. TODO: remove binedges from AxisDescriptor

   sycl::buffer<double, 1>         *fBCoords;           ///< 1D buffer with bufferSize #Dim-dimensional coordinates to fill.
   sycl::buffer<double, 1>         *fBWeights;          ///< Buffer of weigths for each bin on the Host.
   sycl::buffer<int, 1>            *fBBins;             ///< Pointer to array of bins (corresponding to the coordinates) to fill on the GPU.

   int                              fEntries;           ///< Number of entries that have been filled.
   sycl::buffer<double, 1>         *fBStats;            ///< Pointer to statistics array on GPU.

   // Kernel size parameters
   unsigned int                     fNumBlocks;         ///< Number of blocks used in SYCL kernels
   unsigned int                     fBufferSize;        ///< Number of coordinates to buffer.
   unsigned int                     fMaxSmemSize;       ///< Maximum shared memory size per block on device 0.
   unsigned int const               kStatsSmemSize;     ///< Size of shared memory per block in GetStatsKernel
   unsigned int                     fHistoSmemSize;     ///< Size of shared memory per block in HistoKernel
   // clang-format on

public:
   RHnSYCL() = delete;

   RHnSYCL(std::array<int, Dim> ncells, std::array<double, Dim> xlow, std::array<double, Dim> xhigh,
           const double **binEdges);

   ~RHnSYCL()
   {
      delete fBHistogram;
      delete fBAxes;
      delete fDBinEdges;
      delete fBCoords;
      delete fBWeights;
      delete fBBins;
      delete fBStats;
   }

   int GetEntries() const { return fEntries; }

   void RetrieveResults(T *histResult, double *statsResult);

   void Fill(const std::array<double, Dim> &coords, double w = 1.);

private:
   void GetStats(unsigned int size);

   void ExecuteSYCLHisto();
};

} // namespace Experimental
} // namespace ROOT
#endif
