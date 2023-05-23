#ifndef RHnAxis_H
#define RHnAxis_H

namespace ROOT {
namespace Experimental {
struct RAxis {
   int fNbins;  ///< Number of bins(1D) WITH u/overflow
   double fMin; ///< Low edge of first bin
   double fMax; ///< Upper edge of last bin

   const double *kBinEdges; ///< Bin edges array, can be NULL
};

} // namespace Experimental
} // namespace ROOT

#endif