# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(CMAKE_Fortran_COMPILER)
  add_subdirectory(minicern)
endif()

add_subdirectory(gdbPrinters)
add_subdirectory(rmkdepend)
add_subdirectory(win)
