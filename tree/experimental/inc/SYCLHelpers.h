#ifndef SYCL_HELPERS_H
#define SYCL_HELPERS_H

#include <CL/sycl.hpp>
#include "TError.h"
#include <iostream>

namespace sycl = cl::sycl;

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
class InitializeZero {
public:
   InitializeZero(Acc _acc) : acc(_acc) {}

   void operator()(sycl::item<1> item)
   {
      size_t id = item.get_linear_id();
      acc[id] = 0;
   }

private:
   Acc acc;
};

// For debugging.
template <typename Acc>
class PrintArray {
public:
   PrintArray(sycl::stream _out, Acc _acc, const char *_name) : out(_out), acc(_acc), name(_name) {}

   void operator()()
   {
         out << name << "\n";
         for (auto i = 0U; i < acc.size(); i++) {
            out << acc[i] << " ";
         } out << "\n";
   }

private:
   sycl::stream out;
   Acc acc;
   const char *name;
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

} // namespace SYCLHelpers
} // namespace Experimental
} // namespace ROOT
#endif
