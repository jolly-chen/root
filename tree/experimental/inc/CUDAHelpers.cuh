/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <string>
#include <type_traits>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

#include "TError.h"

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      std::stringstream errMsg;
      errMsg << func << "(), " << file + ":" << std::to_string(line) << " : " << cudaGetErrorString(error);
      throw std::runtime_error(errMsg.str());
   }
}

namespace ROOT {
namespace Experimental {
namespace CUDAHelpers {

// Dynamic shared memory needs to be declared as "extern" in CUDA. Having templated kernels with shared memory
// of different data types results in a redeclaration error if the name of the array is the same, so we use a
// proxy function to initialize shared memory arrays of different types with different names.

template <typename T>
__device__ T *shared_memory_proxy()
{
   Fatal("template <typename T> __device__ T *shared_memory_proxy()", "Unsupported shared memory type");
   return (T *)0;
};

template <>
__device__ short *shared_memory_proxy<short>()
{
   extern __shared__ short s_short[];
   return s_short;
}

template <>
__device__ int *shared_memory_proxy<int>()
{
   extern __shared__ int s_int[];
   return s_int;
}

template <>
__device__ double *shared_memory_proxy<double>()
{
   extern __shared__ double s_double[];
   return s_double;
}

template <>
__device__ float *shared_memory_proxy<float>()
{
   extern __shared__ float s_float[];
   return s_float;
}

////////////////////////////////////////////////////////////////////////////////
/// Reduction operations.
template <typename T = double>
struct Plus {
   // Function call operator. The return value is <tt>lhs + rhs</tt>.
   __host__ __device__ constexpr T operator()(const T &lhs, const T &rhs) const { return lhs + rhs; }
};

struct Identity {};

struct Square
{
   __host__ __device__
   double operator ()(const double x) { return x * x; }
};

struct Mul
{
   __host__ __device__
   double operator ()(const double x, const double y) { return x * y; }
};

struct MulSquare
{
   __host__ __device__
   double operator ()(const double x, const double y) { return x * y * y; }
};

struct Mul3
{
   __host__ __device__
   double operator ()(const double x, const double y, const double z) { return x * y * z; }
};


////////////////////////////////////////////////////////////////////////////////
/// CUDA Kernels

// clang-format off
template <unsigned int BlockSize, typename T, typename Op>
__device__ inline void UnrolledReduce(T *sdata, unsigned int tid, Op operation)
{
   // 1024 is the maximum number of threads per block in an NVIDIA GPU:
   // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
   if (BlockSize >= 1024 && tid < 512) { sdata[tid] = operation(sdata[tid], sdata[tid + 512]); } __syncthreads();
   if (BlockSize >= 512  && tid < 256) { sdata[tid] = operation(sdata[tid], sdata[tid + 256]); } __syncthreads();
   if (BlockSize >= 256  && tid < 128) { sdata[tid] = operation(sdata[tid], sdata[tid + 128]); } __syncthreads();
   if (BlockSize >= 128  && tid < 64) { sdata[tid] = operation(sdata[tid], sdata[tid + 64]); } __syncthreads();

   // Reduction within a warp
   if (BlockSize >= 64 && tid < 32) { sdata[tid] = operation(sdata[tid], sdata[tid + 32]); } __syncthreads();
   if (BlockSize >= 32 && tid < 16) { sdata[tid] = operation(sdata[tid], sdata[tid + 16]); } __syncthreads();
   if (BlockSize >= 16 && tid < 8) { sdata[tid] = operation(sdata[tid], sdata[tid + 8]); } __syncthreads();
   if (BlockSize >= 8  && tid < 4) { sdata[tid] = operation(sdata[tid], sdata[tid + 4]); } __syncthreads();
   if (BlockSize >= 4  && tid < 2) { sdata[tid] = operation(sdata[tid], sdata[tid + 2]); } __syncthreads();
   if (BlockSize >= 2  && tid < 1) { sdata[tid] = operation(sdata[tid], sdata[tid + 1]); } __syncthreads();
}
// clang-format on

template <typename T, typename Op>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T r, Op operation)
{
   for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      r = operation(r, __shfl_down_sync(mask, r, offset));
   }
  return r;
}

// See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//     https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu
// The type of reduction in this kernel can be customized by passing different lambda operations
// as transformOp and reduceOp.
// The transformOp lambda function transforms the input values before the reduction step. The transformation can
// involve a variable number of input buffers.
// The reduceOp lambda function should take as input (T lhs, T rhs) and its the return value will be
// written to lhs.
template <unsigned int BlockSize, bool NIsPow2, bool Overwrite, typename TOut, typename TInit, typename TOp, typename ROp, typename...TIn>
__global__ void TransformReduceKernel(unsigned int n, TOut *out, TInit init, TOp transformOp, ROp reduceOp, TIn...in)
{
   volatile TOut *sdata = CUDAHelpers::shared_memory_proxy<TOut>();

   // perform first level of reduction,
   // reading from global memory, writing to shared memory
   unsigned int tid = threadIdx.x;
   unsigned int gridSize = BlockSize * gridDim.x;
   unsigned int maskLength = (BlockSize & 31);  // 31 = warpSize-1
   maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
   const unsigned int mask = (0xffffffff) >> maskLength;

   TOut r;
   if constexpr(std::is_pointer_v<TInit>) {
      r = (TOut) *init;
   } else {
      r = (TOut) init;
   }

   // we reduce multiple elements per thread.  The number is determined by the
   // number of active thread blocks (via gridDim).  More blocks will result
   // in a larger gridSize and therefore fewer elements per thread
   if constexpr(NIsPow2) {
      unsigned int i = blockIdx.x * BlockSize * 2 + threadIdx.x;
      gridSize = gridSize << 1;

      while (i < n) {
         if constexpr(std::is_same_v<TOp, CUDAHelpers::Identity>) {
            r = reduceOp(r, (in[i])...);
            // ensure we don't read out of bounds -- this is optimized away for
            // powerOf2 sized arrays
            if ((i + BlockSize) < n) {
               r = reduceOp(r, (in[i + BlockSize])...);
            }
         } else {
            r = reduceOp(r, transformOp((in[i])...));
            // ensure we don't read out of bounds -- this is optimized away for
            // powerOf2 sized arrays
            if ((i + BlockSize) < n) {
               r = reduceOp(r, transformOp((in[i + BlockSize])...));
            }
         }

         i += gridSize;
      }
   } else {
      unsigned int i = blockIdx.x * BlockSize + threadIdx.x;
      while (i < n) {
         if constexpr(std::is_same_v<TOp, CUDAHelpers::Identity>) {
            r = reduceOp(r, (in[i])...);
         } else {
            r = reduceOp(r, transformOp((in[i])...));
         }

         i += gridSize;
      }
   }

   r = warpReduceSum<TOut>(mask, r, reduceOp);

   if ((tid % warpSize) == 0) {
      sdata[tid / warpSize] = r;
   }

   __syncthreads();

   const unsigned int shmem_extent = (BlockSize / warpSize) > 0 ? (BlockSize / warpSize) : 1;
   const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
   if (tid < shmem_extent) {
      r = sdata[tid];
      r = warpReduceSum<TOut>(ballot_result, r, reduceOp);
   }

   // write result for this block to global mem
   if (tid == 0) {
      if constexpr(Overwrite) {
         out[blockIdx.x] = r;
      } else {
         out[blockIdx.x] = reduceOp(out[blockIdx.x], r);
      }
   }
}

template <typename TOut, typename TOp, typename TInit, typename ROp, typename... TIn>
void TransformReduce(cudaStream_t &stream, std::size_t numBlocks, std::size_t blockSize, std::size_t n,
                     TOut *out, TInit init, bool overwrite, ROp reduceOp, TOp transformOp, TIn ...in)
{
   auto smemSize = ((blockSize / 32) + 1) * sizeof(double);
   bool nIsPow2 = !(n & n-1);

   if (nIsPow2) {
      if (overwrite) {
         if (blockSize == 1)
            TransformReduceKernel<1, true, true><<<numBlocks, 1, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 2)
            TransformReduceKernel<2, true, true><<<numBlocks, 2, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 4)
            TransformReduceKernel<4, true, true><<<numBlocks, 4, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 8)
            TransformReduceKernel<8, true, true><<<numBlocks, 8, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 16)
            TransformReduceKernel<16, true, true><<<numBlocks, 16, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 32)
            TransformReduceKernel<32, true, true><<<numBlocks, 32, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 64)
            TransformReduceKernel<64, true, true><<<numBlocks, 64, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 128)
            TransformReduceKernel<128, true, true><<<numBlocks, 128, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 256)
            TransformReduceKernel<256, true, true><<<numBlocks, 256, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 512)
            TransformReduceKernel<512, true, true><<<numBlocks, 512, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 1024)
            TransformReduceKernel<1024, true, true><<<numBlocks, 1024, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else
            Error("TransformReduce", "Unsupported block size: %lu", blockSize);
      } else {
         if (blockSize == 1)
            TransformReduceKernel<1, true, false><<<numBlocks, 1, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 2)
            TransformReduceKernel<2, true, false><<<numBlocks, 2, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 4)
            TransformReduceKernel<4, true, false><<<numBlocks, 4, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 8)
            TransformReduceKernel<8, true, false><<<numBlocks, 8, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 16)
            TransformReduceKernel<16, true, false><<<numBlocks, 16, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 32)
            TransformReduceKernel<32, true, false><<<numBlocks, 32, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 64)
            TransformReduceKernel<64, true, false><<<numBlocks, 64, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 128)
            TransformReduceKernel<128, true, false><<<numBlocks, 128, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 256)
            TransformReduceKernel<256, true, false><<<numBlocks, 256, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 512)
            TransformReduceKernel<512, true, false><<<numBlocks, 512, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 1024)
            TransformReduceKernel<1024, true, false><<<numBlocks, 1024, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else
            Error("TransformReduce", "Unsupported block size: %lu", blockSize);
      }
   } else {
      if (overwrite) {
         if (blockSize == 1)
            TransformReduceKernel<1, false, true><<<numBlocks, 1, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 2)
            TransformReduceKernel<2, false, true><<<numBlocks, 2, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 4)
            TransformReduceKernel<4, false, true><<<numBlocks, 4, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 8)
            TransformReduceKernel<8, false, true><<<numBlocks, 8, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 16)
            TransformReduceKernel<16, false, true><<<numBlocks, 16, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 32)
            TransformReduceKernel<32, false, true><<<numBlocks, 32, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 64)
            TransformReduceKernel<64, false, true><<<numBlocks, 64, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 128)
            TransformReduceKernel<128, false, true><<<numBlocks, 128, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 256)
            TransformReduceKernel<256, false, true><<<numBlocks, 256, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 512)
            TransformReduceKernel<512, false, true><<<numBlocks, 512, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 1024)
            TransformReduceKernel<1024, false, true><<<numBlocks, 1024, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else
            Error("TransformReduce", "Unsupported block size: %lu", blockSize);
      } else {
         if (blockSize == 1)
            TransformReduceKernel<1, false, false><<<numBlocks, 1, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 2)
            TransformReduceKernel<2, false, false><<<numBlocks, 2, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 4)
            TransformReduceKernel<4, false, false><<<numBlocks, 4, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 8)
            TransformReduceKernel<8, false, false><<<numBlocks, 8, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 16)
            TransformReduceKernel<16, false, false><<<numBlocks, 16, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 32)
            TransformReduceKernel<32, false, false><<<numBlocks, 32, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 64)
            TransformReduceKernel<64, false, false><<<numBlocks, 64, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 128)
            TransformReduceKernel<128, false, false><<<numBlocks, 128, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 256)
            TransformReduceKernel<256, false, false><<<numBlocks, 256, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 512)
            TransformReduceKernel<512, false, false><<<numBlocks, 512, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else if (blockSize == 1024)
            TransformReduceKernel<1024, false, false><<<numBlocks, 1024, smemSize, stream>>>(n, out, init, transformOp, reduceOp, in...);
         else
            Error("TransformReduce", "Unsupported block size: %lu", blockSize);
      }
   }
}

// CUDA version of TMath::BinarySearch
template <typename T>
__device__ Long64_t BinarySearch(Long64_t n, const T *array, T value)
{
   const T *pind;

   pind = thrust::lower_bound(thrust::seq, array, array + n, value);

   if ((pind != array + n) && (*pind == value))
      return (pind - array);
   else
      return (pind - array - 1);

   // return pind - array - !((pind != array + n) && (*pind == value)); // OPTIMIZATION: is this better?
}

// For debugging
__global__ void PrintArray(double *array, int n)
{
   for (int i = 0; i < n; i++) {
      printf("%f ", array[i]);
   }
   printf("\n");
}

} // namespace CUDAHelpers
} // namespace Experimental
} // namespace ROOT
#endif
