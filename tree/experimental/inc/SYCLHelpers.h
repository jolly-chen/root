#ifndef SYCL_HELPERS_H
#define SYCL_HELPERS_H

#include "CL/sycl.hpp"
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

} // namespace SYCLHelpers
} // namespace Experimental
} // namespace ROOT
#endif
