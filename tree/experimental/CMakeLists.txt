# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT tree/experimental package
# @author Jolly Chen, CERN
############################################################################

if (cuda)
  ROOT_LINKER_LIBRARY(CUDAHist src/RHnCUDA.cu TYPE SHARED DEPENDENCIES MathCore)
  target_compile_options(CUDAHist PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(CUDAHist PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(CUDAHist PUBLIC ROOT_RDF_CUDA)
  endif()

if (sycl)
  ROOT_LINKER_LIBRARY(SYCLHist src/RHnSYCL.cxx TYPE SHARED DEPENDENCIES GenVector)
  target_compile_definitions(SYCLHist PUBLIC ROOT_RDF_SYCL)
  list(APPEND COMPUTECPP_USER_FLAGS -no-serial-memop -DCL_TARGET_OPENCL_VERSION=300)

  if (cuda)
  target_compile_definitions(SYCLHist PUBLIC ROOT_RDF_CUDA)
  list(APPEND COMPUTECPP_USER_FLAGS -DCOMPUTE_CPP_BITCODE=ptx64 -sycl -sycl-target ptx64)
  endif()

  target_compile_options(SYCLHist PUBLIC -Wno-unused-parameter)
  add_sycl_to_target(TARGET SYCLHist SOURCES src/RHnSYCL.cxx)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
