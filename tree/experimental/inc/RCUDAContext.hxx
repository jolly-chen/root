#ifndef RCUDAContext_H
#define RCUDAContext_H

#include <vector>
#include <iostream>

namespace ROOT {
namespace Experimental {
class RCUDAContext {
private:
#ifdef __CUDACC__
   std::vector<cudaStream_t> fStreams;
#endif

   int fMaxSMem;            // Shared memory capacity per block
   int fMaxThreadsPerBlock; // Maximum number of threads per block
   int fMaxBlockDimX;       // Maximum x-dimension of a block

   static int fNStreams;

public:
   RCUDAContext(RCUDAContext &other) = delete;
   void operator=(const RCUDAContext &) = delete;

   static inline RCUDAContext &instance()
   {
      static RCUDAContext ctx;
      return ctx;
   }

   inline static void SetStreams(int nStreams) {
      fNStreams = nStreams;
      if (getenv("DBG")) {
         printf("Set number of CUDA streams to %d\n", nStreams);
      }
   }

   inline int GetMaxSMem() { return fMaxSMem; }
   inline int GetMaxThreadsPerBlock() { return fMaxThreadsPerBlock; }
   inline int GetMaxBlockDimX() { return fMaxBlockDimX; }

#ifdef __CUDACC__
   cudaStream_t &GetStream(int i);
#endif

protected:
   RCUDAContext();
   ~RCUDAContext();
};

} // namespace Experimental
} // namespace ROOT

#endif
