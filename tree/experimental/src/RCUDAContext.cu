#include "RCUDAContext.hxx"

#include <vector>
#include <iostream>

namespace ROOT {
namespace Experimental {

int RCUDAContext::fNStreams = 2;

RCUDAContext::RCUDAContext() : fStreams(fNStreams)
{
   for (int i = 0; i < fNStreams; i++)
      cudaStreamCreate(&fStreams[i]);

   cudaDeviceGetAttribute(&fMaxSMem, cudaDevAttrMaxSharedMemoryPerBlock, 0);
   cudaDeviceGetAttribute(&fMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
   cudaDeviceGetAttribute(&fMaxBlockDimX, cudaDevAttrMaxBlockDimX, 0);
}

RCUDAContext::~RCUDAContext()
{
   for (int i = 0; i < fNStreams; i++)
      cudaStreamDestroy(fStreams[i]);
}

cudaStream_t &RCUDAContext::GetStream(int i)
{
   if (i >= RCUDAContext::fNStreams) {
      fprintf(stderr, "CUDA Stream %d does not exist. Using stream 0.\n", i);
      return fStreams[0];
   }
   return fStreams[i];
}

} // namespace Experimental
} // namespace ROOT