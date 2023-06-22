#ifndef SYCL_HELPERS_H
#define SYCL_HELPERS_H

#include <sycl/sycl.hpp>
#include "TError.h"
#include <iostream>

namespace ROOT {
namespace Experimental {
namespace SYCLHelpers {

#define ERRCHECK(code)                                    \
   {                                                      \
      try {                                               \
         code;                                            \
         queue.wait_and_throw();                          \
      } catch (sycl::exception const &e) {                \
         std::cout << "Caught synchronous SYCL exception" \
                   << " :" << __LINE__ << ":\n"           \
                   << e.what() << std::endl;              \
      }                                                   \
   }

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
class InitializeToZeroTask {
public:
   InitializeToZeroTask(Acc _acc) : acc(_acc) {}

   void operator()(sycl::id<1> id) const { acc[id] = 0; }

private:
   Acc acc;
};

// Can't use std::lower_bound on GPU so we define it here...
template <class ForwardIt, class T>
const T *lower_bound(ForwardIt first, ForwardIt last, const T &val)
{
   ForwardIt it;
   int step;
   int count = last - first;

   while (count > 0) {
      it = first;
      step = count / 2;
      it += step;

      if (*it < val) {
         first = ++it;
         count -= step + 1;
      } else {
         count = step;
      }
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
void InitializeToZero(sycl::queue &queue, T arr, size_t n)
{
   queue.submit([&](sycl::handler &cgh) {
      sycl::accessor acc{arr, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for(sycl::range<1>(n), SYCLHelpers::InitializeToZeroTask(acc));
   });
}

// For debugging...
template <class T>
void PrintArray(sycl::queue &queue, T arr, size_t n)
{
   try {
      queue
         .submit([&](sycl::handler &cgh) {
            sycl::stream out(1024, 256, cgh);
            auto acc = arr->template get_access<sycl::access::mode::read>(cgh);
            cgh.single_task([=]() {
               for (auto i = 0U; i < n; i++) {
                  out << acc[i] << " ";
               }
               out << "\n";
            });
         })
         .wait_and_throw();
   } catch (sycl::exception const &e) {
      std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
   }
}

template <class T>
void PrintVar(sycl::queue &queue, T &var)
{
   try {
      queue
         .submit([&](sycl::handler &cgh) {
            sycl::stream out(1024, 256, cgh);
            cgh.single_task([=]() { out << var << "\n"; });
         })
         .wait_and_throw();
   } catch (sycl::exception const &e) {
      std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
   }
}

} // namespace SYCLHelpers
} // namespace Experimental
} // namespace ROOT
#endif
