#ifndef RHnCUDA_H
#define RHnCUDA_H

#include <vector>
#include <array>
#include "ROOT/RVec.hxx"
#include "RCUDAContext.hxx"

namespace ROOT {
namespace Experimental {

template <typename T, unsigned int Dim, unsigned int BlockSize = 256>
class RHnCUDA {
   // clang-format off
private:
   static constexpr int kNStats = 2 + Dim * 2 + Dim * (Dim - 1) / 2; ///< Number of statistics.
   static constexpr int kMaxBlocks = 65536 / BlockSize;              ///< Threshold for reductions

   T                                *fDHistogram;         ///< Pointer to histogram buffer on the GPU.
   int                               fNBins;              ///< Total number of bins in the histogram WITH under/overflow

   int                              *fDNBinsAxis;         ///< Number of bins(1D) WITH u/overflow per axis
   double                           *fDMin;               ///< Low edge of first bin per axis
   double                           *fDMax;               ///< Upper edge of last bin per axis
   double                           *fDBinEdges;          ///< Bin edges array for each axis
   int                              *fDBinEdgesIdx;       ///< Start index of the binedges in kBinEdges per axis

   double                           *fHCoords;            ///<
   double                           *fHWeights;           ///<
   double                           *fDCoords;            ///< Pointer to array of coordinates to fill on the GPU.
   double                           *fDWeights;           ///< Pointer to array of weights on the GPU.
   bool                             *fDMask;              ///< Pointer to array of bins (corresponding to the coordinates) to fill on the GPU.

   std::size_t                       fEntries;            ///< Number of entries that have been filled.
   double                           *fDIntermediateStats; ///< Buffer for storing intermediate results of stat reduction on GPU.
   double                           *fDStats;             ///< Pointer to statistics array on GPU.

   // Kernel parameters
   std::size_t                       fMaxBulkSize;        ///< Number of coordinates to buffer.
   const std::size_t                 kStatsSmemSize;      ///< Size of shared memory per block in GetStatsKernel
   std::size_t                       fHistoSmemSize;      ///< Size of shared memory per block in HistoKernel

   RCUDAContext                     &fCtx;
   // clang-format on

public:
   RHnCUDA() = delete;

   // TODO: Change RHnCUDA to SOA for axes
   RHnCUDA(std::size_t maxBulkSize, const std::size_t nBins,
           const std::array<int, Dim> &nBinsAxis, const std::array<double, Dim> &xLow,
           const std::array<double, Dim> &xHigh, const std::vector<double> &binEdges,
           const std::array<int, Dim> &binEdgesIdx);

   ~RHnCUDA();

   RHnCUDA(const RHnCUDA &) = delete;
   RHnCUDA &operator=(const RHnCUDA &) = delete;

   std::size_t GetEntries() { return fEntries; }

   void RetrieveResults(int numStream, T *histResult, double *statsResult);

   void Fill(int numStream, const RVecD &coords);

   void Fill(int numStream, const RVecD &coords, const RVecD &weights);

   size_t GetMaxBulkSize() { return fMaxBulkSize; }

protected:
#ifdef __CUDACC__
   void GetNumBlocksAndThreads(std::size_t n, int &blocks, int &threads);

   void GetStats(cudaStream_t &stream, std::size_t size);

   void ExecuteCUDAHisto(cudaStream_t &stream, std::size_t size);
#endif
};

} // namespace Experimental
} // namespace ROOT
#endif
