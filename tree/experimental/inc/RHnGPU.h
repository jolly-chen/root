#ifndef RHnGPU_H
#define RHnGPU_H

namespace ROOT {
namespace Experimental {
struct RAxis {
   int fNbins;  ///< Number of bins(1D) WITH u/overflow
   double fMin; ///< Low edge of first bin
   double fMax; ///< Upper edge of last bin

   const double *kBinEdges; ///< Bin edges array, can be NULL
};

template <typename, unsigned int, unsigned int = 256>
class RHnCUDA;

template <typename, unsigned int, unsigned int = 256>
class RHnSYCL;
} // namespace Experimental
} // namespace ROOT

#endif