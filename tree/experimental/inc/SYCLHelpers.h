#ifndef SYCL_HELPERS_H
#define SYCL_HELPERS_H

#include <sycl/sycl.hpp>
#include "TError.h"
#include <iostream>

namespace ROOT {
namespace Experimental {
namespace SYCLHelpers {

auto exception_handler(sycl::exception_list exceptions)
{
   for (std::exception_ptr const &e_ptr : exceptions) {
      try {
         std::rethrow_exception(e_ptr);
      } catch (sycl::exception const &e) {
         std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
      }
   }
}

template <typename Acc>
class InitializeZeroTask {
public:
   InitializeZeroTask(Acc _acc) : acc(_acc) {}

   void operator()(sycl::item<1> item)
   {
      size_t id = item.get_linear_id();
      acc[id] = 0;
   }

private:
   Acc acc;
};

// Can't use std::lower_bound on GPU so we define it here...
template <typename T>
const T *lower_bound(const T *first, const T *last, T val)
{
   size_t len = last - first;

   while (len > 0) {
      auto half = len >> 1;
      auto middle = first + half;

      if (*middle < val) {
         first = middle;
         ++first;
         len -= half - 1;
      } else
         len = half;
   }
   return first;
}

template <typename T>
long long BinarySearch(long long n, const T *array, T value)
{
   auto pind = lower_bound(array, array + n, value);

   if ((pind != array + n) && (*pind == value))
      return (pind - array);
   else
      return (pind - array - 1);
}

template <typename T>
void InitializeZero(sycl::queue &queue, T arr, size_t n)
{
   queue.submit([&](sycl::handler &cgh) {
      sycl::stream out(1024, 256, cgh);

      auto acc = arr->get_access<sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for(sycl::range<1>(n), SYCLHelpers::InitializeZeroTask(acc));
   });
}

#ifdef DEBUG
template <class T>
void PrintArray(sycl::queue &queue, T &arr)
{
   queue.submit([&](sycl::handler &cgh) {
      sycl::stream out(1024, 256, cgh);
      auto acc = arr->template get_access<sycl::access::mode::read>(cgh);
      cgh.single_task([=]() {
         for (auto i = 0U; i < 1; i++) {
            out << acc[i] << " ";
         }
         out << "\n";
      });
   });
   queue.wait_and_throw();
}

template <class T>
void PrintVar(sycl::queue &queue, T &var)
{
   queue.submit([&](sycl::handler &cgh) {
      sycl::stream out(1024, 256, cgh);
      cgh.single_task([=]() { out << var << "\n"; });
   });
   queue.wait_and_throw();
}
#endif

} // namespace SYCLHelpers
} // namespace Experimental
} // namespace ROOT
#endif
