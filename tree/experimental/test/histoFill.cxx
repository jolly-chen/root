////////////////////////////////////////////////////////////////////////////////////
/// Tests for filling RHnCUDA histograms with different data types and dimensions.
///
#include <climits>
#include <stdlib.h>
#include "gtest/gtest.h"

#include "ROOT/RDataFrame.hxx"
#include "TH1.h"
#include "TAxis.h"

#ifdef ROOT_RDF_CUDA
#include "RHnCUDA.h"
using ROOT::Experimental::RHnCUDA;
struct CUDAHist {
   template <typename T, unsigned int Dim>
   using type = RHnCUDA<T, Dim>;

   static constexpr int histIdx = 0;
};
#endif

#ifdef ROOT_RDF_SYCL
#include "RHnSYCL.h"
using ROOT::Experimental::RHnSYCL;
struct SYCLHist {
   template <typename T, unsigned int Dim>
   using type = RHnSYCL<T, Dim>;

   static constexpr int histIdx = 1;
};
#endif

#if defined(ROOT_RDF_CUDA) && defined(ROOT_RDF_SYCL)
std::vector<const char *> test_environments = {"CUDA_HIST", "SYCL_HIST"};
#elif defined(ROOT_RDF_CUDA)
std::vector<const char *> test_environments = {"CUDA_HIST"};
#elif defined(ROOT_RDF_SYCL)
std::vector<const char *> test_environments = {"SYCL_HIST"};
#endif

/**
 * Helper functions for toggling ON/OFF GPU histogramming.
 */

void DisableGPU()
{
   for (unsigned int i = 0; i < test_environments.size(); i++)
      unsetenv(test_environments[i]);
}

void EnableGPU(const char *env)
{
   DisableGPU();
   setenv(env, "1", 1);
}

// Returns an array with the given value repeated n times.
template <typename T, int n>
std::array<T, n> Repeat(T val)
{
   std::array<T, n> result;
   result.fill(val);
   return result;
}

// Helper functions for element-wise comparison of histogram arrays.
#define CHECK_ARRAY(a, b, n)                              \
   {                                                      \
      for (auto i : ROOT::TSeqI(n)) {                     \
         EXPECT_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                   \
   }

#define CHECK_ARRAY_FLOAT(a, b, n)                              \
   {                                                            \
      for (auto i : ROOT::TSeqI(n)) {                           \
         EXPECT_FLOAT_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                         \
   }

#define CHECK_ARRAY_DOUBLE(a, b, n)                              \
   {                                                             \
      for (auto i : ROOT::TSeqI(n)) {                            \
         EXPECT_DOUBLE_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                          \
   }

template <typename T>
void CompareArrays(T *result, T *expected, int n)
{
   CHECK_ARRAY(result, expected, n)
}

template <>
void CompareArrays(float *result, float *expected, int n)
{
   CHECK_ARRAY_FLOAT(result, expected, n)
}

template <>
void CompareArrays(double *result, double *expected, int n)
{
   CHECK_ARRAY_DOUBLE(result, expected, n)
}

// Create all combinations of datatype and dimension
// Partially taken from https://stackoverflow.com/questions/56115790/gtest-parametrized-tests-for-different-types

// Test "parameters"
struct OneDim {
   static constexpr unsigned int dim = 1;
};
struct TwoDim {
   static constexpr unsigned int dim = 2;
};
struct ThreeDim {
   static constexpr unsigned int dim = 3;
};

template <typename T, class Dim, class Hist>
struct Case {
   using dataType = T;
   using histType = typename Hist::template type<T, Dim::dim>;

   static constexpr int GetDim() { return Dim::dim; }
   static constexpr const char *GetEnv() { return test_environments[Hist::histIdx]; }
};

template <class TupleType, class TupleDims, class TupleHists, std::size_t I>
struct make_case {
   static constexpr std::size_t N = std::tuple_size<TupleDims>::value;
   static constexpr std::size_t M = std::tuple_size<TupleHists>::value;
   using type = Case<typename std::tuple_element<I / (N * M), TupleType>::type,
                     typename std::tuple_element<(I / M) % N, TupleDims>::type,
                     typename std::tuple_element<I % M, TupleHists>::type>;
};

template <class T1, class T2, class T3, class Is>
struct make_combinations;

template <class TupleType, class TupleDims, class TupleHists, std::size_t... Is>
struct make_combinations<TupleType, TupleDims, TupleHists, std::index_sequence<Is...>> {
   using tuples = std::tuple<typename make_case<TupleType, TupleDims, TupleHists, Is>::type...>;
};

template <class TupleTypes, class TupleDims, class TupleHists>
using Combinations_t = typename make_combinations<
   TupleTypes, TupleDims, TupleHists,
   std::make_index_sequence<std::tuple_size<TupleTypes>::value * std::tuple_size<TupleDims>::value *
                            std::tuple_size<TupleHists>::value>>::tuples;

template <typename C>
class FillTestFixture : public ::testing::Test {
protected:
   // Includes u/overflow bins. Uneven number chosen to have a center bin.
   const static int numBins = 7;

   // Variables for defining fixed bins.
   const double startBin = 1;
   const double endBin = 4;

   // int, double, float
   using dataType = typename C::dataType;

   // 1, 2, or 3
   static constexpr int dim = C::GetDim();

   // cuda or sycl
   const char *env = C::GetEnv();

   // Total number of cells
   const static int nCells = pow(numBins, dim);
   dataType result[nCells], expectedHist[nCells];

   double *stats, *expectedStats;
   int nStats;

   typename C::histType histogram;

   FillTestFixture()
      : histogram(32768, nCells, Repeat<int, dim>(numBins), Repeat<double, dim>(startBin), Repeat<double, dim>(endBin),
                  {}, Repeat<int, dim>(-1))
   {
   }

   void SetUp() override
   {
      EnableGPU(env);
      nStats = 2 + dim * 2 + dim * (dim - 1) / 2;

      stats = new double[nStats];
      expectedStats = new double[nStats];

      memset(stats, 0, nStats * sizeof(double));
      memset(expectedStats, 0, nStats * sizeof(double));
      memset(expectedHist, 0, nCells * sizeof(dataType));
   }

   void TearDown() override { delete[] stats; }

   bool UOverflow(ROOT::RVecD coord)
   {
      for (auto d = 0; d < dim; d++) {
         if (coord[d] < startBin || coord[d] > endBin)
            return true;
      }
      return false;
   }

   void GetExpectedStats(std::vector<ROOT::RVecD> coords, dataType weight)
   {
      for (auto i = 0; i < (int)coords.size(); i++) {
         if (UOverflow(coords[i]))
            continue;

         // Tsumw
         expectedStats[0] += weight;
         // Tsumw2
         expectedStats[1] += weight * weight;

         auto offset = 2;
         for (auto d = 0; d < dim; d++) {
            // e.g. Tsumwx
            expectedStats[offset++] += weight * coords[i][d];
            // e.g. Tsumwx2
            expectedStats[offset++] += weight * pow(coords[i][d], 2);

            for (auto prev_d = 0; prev_d < d; prev_d++) {
               // e.g. Tsumwxy
               this->expectedStats[offset++] += weight * coords[i][d] * coords[i][prev_d];
            }
         }
      }
   }
};

template <typename T>
struct Test;

template <typename... T>
struct Test<std::tuple<T...>> {
   using Types = ::testing::Types<T...>;
};

#if defined(ROOT_RDF_CUDA) && defined(ROOT_RDF_SYCL)
using FillTestTypes = Test<Combinations_t<std::tuple<double, float, int, short>, std::tuple<OneDim, TwoDim, ThreeDim>,
                                          std::tuple<CUDAHist, SYCLHist>>>::Types;
#elif defined(ROOT_RDF_CUDA)
using FillTestTypes = Test<Combinations_t<std::tuple<double, float, int, short>, std::tuple<OneDim, TwoDim, ThreeDim>,
                                          std::tuple<CUDAHist>>>::Types;
#elif defined(ROOT_RDF_SYCL)
using FillTestTypes = Test<Combinations_t<std::tuple<double, float, int, short>, std::tuple<OneDim, TwoDim, ThreeDim>,
                                          std::tuple<SYCLHist>>>::Types;
#endif
TYPED_TEST_SUITE(FillTestFixture, FillTestTypes);

template <class Hist>
class ClampTestFixture : public ::testing::Test {
protected:
   using hist = Hist;
   ClampTestFixture() {}

   void SetUp() override { EnableGPU(test_environments[Hist::histIdx]); }

   void TearDown() override {}
};

#if defined(ROOT_RDF_CUDA) && defined(ROOT_RDF_SYCL)
using ClampTestTypes = ::testing::Types<CUDAHist, SYCLHist>;
#elif defined(ROOT_RDF_CUDA)
using ClampTestTypes = ::testing::Types<CUDAHist>;
#elif defined(ROOT_RDF_SYCL)
using ClampTestTypes = ::testing::Types<SYCLHist>;
#endif

TYPED_TEST_SUITE(ClampTestFixture, ClampTestTypes);

/////////////////////////////////////
/// Test Cases

TYPED_TEST(FillTestFixture, FillFixedBins)
{
   // int, double, or float
   using t = typename TestFixture::dataType;
   auto &h = this->histogram;

   std::vector<ROOT::RVecD> coords = {
      ROOT::RVecD(this->dim, this->startBin - 1),                   // Underflow
      ROOT::RVecD(this->dim, (this->startBin + this->endBin) / 2.), // Center
      ROOT::RVecD(this->dim, this->endBin + 1)                      // OVerflow
   };
   auto weight = (t)1;

   std::vector<int> expectedHistBins = {0, this->nCells / 2, this->nCells - 1};

   for (auto i = 0; i < (int)coords.size(); i++) {
      h.Fill(0, coords[i]);
      this->expectedHist[expectedHistBins[i]] = weight;
   }

   h.RetrieveResults(0, this->result, this->stats);

   {
      SCOPED_TRACE("Check fill result");
      CompareArrays(this->result, this->expectedHist, this->nCells);
   }

   {
      SCOPED_TRACE("Check statistics");
      this->GetExpectedStats(coords, weight);
      CompareArrays(this->stats, this->expectedStats, this->nStats);
   }
}

TYPED_TEST(FillTestFixture, FillFixedBinsWeighted)
{
   // int, double, or float
   using t = typename TestFixture::dataType;
   auto &h = this->histogram;

   std::vector<ROOT::RVecD> coords = {
      ROOT::RVecD(this->dim, this->startBin - 1),                   // Underflow
      ROOT::RVecD(this->dim, (this->startBin + this->endBin) / 2.), // Center
      ROOT::RVecD(this->dim, this->endBin + 1)                      // OVerflow
   };
   auto weight = ROOT::RVecD(1, 7);

   std::vector<int> expectedHistBins = {0, this->nCells / 2, this->nCells - 1};

   for (auto i = 0; i < (int)coords.size(); i++) {
      h.Fill(0, coords[i], weight);
      this->expectedHist[expectedHistBins[i]] = (t)weight[0];
   }

   h.RetrieveResults(0, this->result, this->stats);

   {
      SCOPED_TRACE("Check fill result");
      CompareArrays(this->result, this->expectedHist, this->nCells);
   }

   {
      SCOPED_TRACE("Check statistics");
      this->GetExpectedStats(coords, (t)weight[0]);
      CompareArrays(this->stats, this->expectedStats, this->nStats);
   }
}

TYPED_TEST(ClampTestFixture, FillIntClamp)
{
   auto h = typename TestFixture::hist::template type<int, 1>(32768, 6, {6}, {0}, {4}, {}, {-1});
   h.Fill(0, {0}, {INT_MAX});
   h.Fill(0, {3}, {-INT_MAX});

   for (int i = 0; i < 100; i++) {       // Repeat to test for race conditions
      h.Fill(0, {0});                       // Should keep max value
      h.Fill(0, {1}, {long(INT_MAX) + 1});  // Clamp positive overflow
      h.Fill(0, {2}, {-long(INT_MAX) - 1}); // Clamp negative overflow
      h.Fill(0, {3}, {-1});                 // Should keep min value
   }

   int result[6];
   double s[4];
   h.RetrieveResults(0, result, s);

   EXPECT_EQ(result[0], 0);
   EXPECT_EQ(result[1], INT_MAX);
   EXPECT_EQ(result[2], INT_MAX);
   EXPECT_EQ(result[3], -INT_MAX);
   EXPECT_EQ(result[4], -INT_MAX);
   EXPECT_EQ(result[5], 0);
}

TYPED_TEST(ClampTestFixture, FillShortClamp)
{
   auto h = typename TestFixture::hist::template type<short, 1>(32768, 10, {10}, {0}, {8}, {}, {-1});

   // Filling short histograms is implemented using atomic operations on integers so we test each case
   // twice to test the for correct filling of the lower and upper bits.
   for (int offset = 0; offset < 2; offset++) {
      h.Fill(0, {0. + offset}, {32767});
      h.Fill(0, {2. + offset}, {-32767});

      for (int i = 0; i < 100; i++) {     // Repeat to test for race conditions
         h.Fill(0, {0. + offset});           // Keep max value
         h.Fill(0, {2. + offset}, {-1});     // Keep min value
         h.Fill(0, {4. + offset}, {32769});  // Clamp positive overflow
         h.Fill(0, {6. + offset}, {-32769}); // Clamp negative overflow
      }
   }

   short result[10];
   double s[4];
   h.RetrieveResults(0, result, s);

   int expected[10] = {0, 32767, 32767, -32767, -32767, 32767, 32767, -32767, -32767, 0};
   CHECK_ARRAY(result, expected, 10);
}
