# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# set (histoRDFLibs ROOTDataFrame)
# set (invariantMassLibs ROOTVecOps Physics)

# if (cuda)
#     ROOT_ADD_GTEST(HistoFill histoFill.cxx LIBRARIES CUDAHist Hist ROOTDataFrame)
#     list(APPEND histoRDFLibs CUDAHist)
# endif()

# if (sycl)
#     list(APPEND histoRDFLibs SYCLHist)
#     list(APPEND invariantMassLibs InvariantMassSYCL)
#     ROOT_ADD_GTEST(InvariantMass invariantMass.cxx LIBRARIES ${invariantMassLibs})
# endif()

# if (sycl OR cuda)
#     ROOT_ADD_GTEST(HistoRDF histoRDF.cxx LIBRARIES ${histoRDFLibs})
# endif()
