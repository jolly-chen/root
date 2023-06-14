/// Template instantations for RHnSYCL.
///
/// The implementation of the template class cannot be added to the header file RHnSYCL.h because this
/// header is included in cpp files to access the SYCL histogramming class. The cpp files are compiled
/// with a cpp compiler which will not be recognize any SYCL calls, so any calls to the SYCL API needs
/// to be separated and only compiled by the nvcc compiler.
/// TODO: support for char and short histograms.

// clang-format off
// template class RHnSYCL<char,   1, 64>;
// template class RHnSYCL<char,   2, 64>;
// template class RHnSYCL<char,   3, 64>;
// template class RHnSYCL<short,  1, 64>;
// template class RHnSYCL<short,  2, 64>;
// template class RHnSYCL<short,  3, 64>;
template class RHnSYCL<int,    1, 64>;
template class RHnSYCL<int,    2, 64>;
template class RHnSYCL<int,    3, 64>;
template class RHnSYCL<float,  1, 64>;
template class RHnSYCL<float,  2, 64>;
template class RHnSYCL<float,  3, 64>;
template class RHnSYCL<double, 1, 64>;
template class RHnSYCL<double, 2, 64>;
template class RHnSYCL<double, 3, 64>;

// template class RHnSYCL<char,   1, 128>;
// template class RHnSYCL<char,   2, 128>;
// template class RHnSYCL<char,   3, 128>;
// template class RHnSYCL<short,  1, 128>;
// template class RHnSYCL<short,  2, 128>;
// template class RHnSYCL<short,  3, 128>;
template class RHnSYCL<int,    1, 128>;
template class RHnSYCL<int,    2, 128>;
template class RHnSYCL<int,    3, 128>;
template class RHnSYCL<float,  1, 128>;
template class RHnSYCL<float,  2, 128>;
template class RHnSYCL<float,  3, 128>;
template class RHnSYCL<double, 1, 128>;
template class RHnSYCL<double, 2, 128>;
template class RHnSYCL<double, 3, 128>;

// template class RHnSYCL<char,   1, 256>;
// template class RHnSYCL<char,   2, 256>;
// template class RHnSYCL<char,   3, 256>;
// template class RHnSYCL<short,  1, 256>;
// template class RHnSYCL<short,  2, 256>;
// template class RHnSYCL<short,  3, 256>;
template class RHnSYCL<int,    1, 256>;
template class RHnSYCL<int,    2, 256>;
template class RHnSYCL<int,    3, 256>;
template class RHnSYCL<float,  1, 256>;
template class RHnSYCL<float,  2, 256>;
template class RHnSYCL<float,  3, 256>;
template class RHnSYCL<double, 1, 256>;
template class RHnSYCL<double, 2, 256>;
template class RHnSYCL<double, 3, 256>;

// template class RHnSYCL<char,   1, 512>;
// template class RHnSYCL<char,   2, 512>;
// template class RHnSYCL<char,   3, 512>;
// template class RHnSYCL<short,  1, 512>;
// template class RHnSYCL<short,  2, 512>;
// template class RHnSYCL<short,  3, 512>;
template class RHnSYCL<int,    1, 512>;
template class RHnSYCL<int,    2, 512>;
template class RHnSYCL<int,    3, 512>;
template class RHnSYCL<float,  1, 512>;
template class RHnSYCL<float,  2, 512>;
template class RHnSYCL<float,  3, 512>;
template class RHnSYCL<double, 1, 512>;
template class RHnSYCL<double, 2, 512>;
template class RHnSYCL<double, 3, 512>;

// template class RHnSYCL<char,   1, 1024>;
// template class RHnSYCL<char,   2, 1024>;
// template class RHnSYCL<char,   3, 1024>;
// template class RHnSYCL<short,  1, 1024>;
// template class RHnSYCL<short,  2, 1024>;
// template class RHnSYCL<short,  3, 1024>;
template class RHnSYCL<int,    1, 1024>;
template class RHnSYCL<int,    2, 1024>;
template class RHnSYCL<int,    3, 1024>;
template class RHnSYCL<float,  1, 1024>;
template class RHnSYCL<float,  2, 1024>;
template class RHnSYCL<float,  3, 1024>;
template class RHnSYCL<double, 1, 1024>;
template class RHnSYCL<double, 2, 1024>;
template class RHnSYCL<double, 3, 1024>;
// clang-format on
