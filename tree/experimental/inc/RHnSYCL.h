#ifndef RHnSYCL_H
#define RHnSYCL_H

#include <vector>
#include <CL/sycl.hpp>

#include <array>
#include <utility>

namespace sycl = cl::sycl;

namespace ROOT {
namespace Experimental {
template <typename T, unsigned int Dim, unsigned int BlockSize>
class RHnSYCL {

   // clang-format off
private:
   sycl::queue              queue;

   sycl::buffer<T, 1>       fHistogram;         ///< Pointer to histogram buffer on the GPU.
   int                      fNbins;             ///< Total number of bins in the histogram WITH under/overflow

   const int                kNStats;            ///< Number of statistics.
   sycl::buffer<RAxis, 1>   fAxes;              ///< Vector of Dim axis descriptors

   sycl::buffer<double, 1>  fCoords;            ///< 1D buffer with bufferSize #Dim-dimensional coordinates to fill.
   sycl::buffer<double, 1>  fWeights;           ///< Buffer of weigths for each bin on the Host.
   sycl::buffer<int, 1>     fBins;              ///< Pointer to array of bins (corresponding to the coordinates) to fill on the GPU.

   int                      fEntries;           ///< Number of entries that have been filled.

   // Kernel size parameters
   unsigned int             fNumBlocks;         ///< Number of blocks used in SYCL kernels
   unsigned int             fBufferSize;        ///< Number of coordinates to buffer.
   unsigned int             fMaxSmemSize;       ///< Maximum shared memory size per block on device 0.
   unsigned int const       kStatsSmemSize;     ///< Size of shared memory per block in GetStatsKernel
   unsigned int             fHistoSmemSize;     ///< Size of shared memory per block in HistoKernel
   // clang-format on

public:
   RHnSYCL() = delete;

   RHnSYCL(std::array<int, Dim> ncells, std::array<double, Dim> xlow, std::array<double, Dim> xhigh,
           const double **binEdges = NULL);

   int GetEntries() const { return fEntries; }

   void RetrieveResults(T *histResult, double *statsResult);

   void Fill(const std::array<double, Dim> &coords, double w = 1.);

private:
   void AllocateBuffers();

   void GetStats(unsigned int size);

   void ExecuteSYCLHisto();
};

} // namespace Experimental
} // namespace ROOT
#endif
