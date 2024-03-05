#ifndef RHnSYCL_H
#define RHnSYCL_H

#include <array>
#include <optional>
#include <sycl/sycl.hpp>
#include "ROOT/RVec.hxx"

namespace ROOT {
namespace Experimental {

template <typename T, unsigned int Dim, unsigned int WGroupSize = 256>
class RHnSYCL {
private:
   // B = SYCL buffer
   // S = SYCL USM shared pointer
   // D = SYCL USM device pointer

   static constexpr int kNStats = 2 + Dim * 2 + Dim * (Dim - 1) / 2; ///< Number of statistics.

   // clang-format off
   sycl::queue                       queue;
   std::vector<sycl::event>          prevBulk;

   T                                *fDHistogram;         ///< Pointer to histogram buffer on the GPU.
   int                               fNBins;              //< Total number of bins in the histogram WITH under/overflow

   int                              *fDNBinsAxis;         ///< Number of bins(1D) WITH u/overflow per axis
   double                           *fDMin;               ///< Low edge of first bin per axis
   double                           *fDMax;               ///< Upper edge of last bin per axis
   double                           *fDBinEdges;          ///< Bin edges array for each axis
   int                              *fDBinEdgesIdx;       ///< Start index of the binedges in kBinEdges per axis

   double                           *fDCoords;            ///< Pointer to array of coordinates to fill on the GPU.
   double                           *fDWeights;           ///< Pointer to array of weights on the GPU.
   int                              *fDBins;              ///< Pointer to array of bins (corresponding to the coordinates) to fill on the GPU.

   int                               fEntries;            ///< Number of entries that have been filled.
   double                           *fDStats;             ///< Pointer to statistics array on GPU.
   double                           *fDIntermediateStats; ///< Pointer to statistics array on GPU.

   // Kernel size parameters
   std::size_t                       fMaxBulkSize;        ///< Number of coordinates to buffer.
   std::size_t                       fMaxSmemSize;        ///< Maximum shared memory size per block on device 0.
   std::size_t const                 kStatsSmemSize;      ///< Size of shared memory per block in GetStatsKernel
   std::size_t                       fHistoSmemSize;      ///< Size of shared memory per block in HistoKernel
   // clang-format on

public:
   RHnSYCL() = delete;

   RHnSYCL(std::size_t maxBulkSize, const std::size_t nBins, const std::array<int, Dim> &nBinsAxis,
           const std::array<double, Dim> &xLow, const std::array<double, Dim> &xHigh,
           const std::vector<double> &binEdges, const std::array<int, Dim> &binEdgesIdx);

   ~RHnSYCL()
   {
      sycl::free(fDHistogram, queue);
      sycl::free(fDNBinsAxis, queue);
      sycl::free(fDMin, queue);
      sycl::free(fDMax, queue);
      sycl::free(fDBinEdgesIdx, queue);
      sycl::free(fDCoords, queue);
      sycl::free(fDWeights, queue);
      sycl::free(fDBins, queue);
      sycl::free(fDStats, queue);
      if (fDIntermediateStats != NULL)
         sycl::free(fDIntermediateStats, queue);
      if (fDBinEdges != NULL)
         sycl::free(fDBinEdges, queue);
   }

   RHnSYCL(const RHnSYCL &) = delete;
   RHnSYCL &operator=(const RHnSYCL &) = delete;

   int GetEntries() const { return fEntries; }

   void RetrieveResults(T *histResult, double *statsResult);

   void Fill(const RVecD &coords);

   void Fill(const RVecD &coords, const RVecD &weights);

   std::size_t GetMaxBulkSize() { return fMaxBulkSize; }

private:
   void GetStats(std::size_t size, sycl::event &fillEvent);

   void ExecuteSYCLHisto(std::size_t size, std::vector<sycl::event> &depends);
};

} // namespace Experimental
} // namespace ROOT
#endif
