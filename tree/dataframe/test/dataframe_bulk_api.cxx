#include "RtypesCore.h"
#include "ROOT/RDF/REventMask.hxx"
#include <ROOT/RDataFrame.hxx>

#include <gtest/gtest.h>
#include <thread>

using ROOT::RDF::Experimental::REventMask;

TEST(RDFBulkAPI, SupportsTreeBulkIO)
{
   TTree t("t", "t");
   int x = 0;
   float arr[2]{}; // RDF thinks that static-size arrays of size 1 are scalars
   t.Branch("x", &x);
   t.Branch("arr", &arr, "arr[2]/F");
   t.Fill();

   auto df = ROOT::RDataFrame(t).Alias("z", "x");

   EXPECT_TRUE(df.SupportsTreeBulkIO<int>("x").first);
   EXPECT_TRUE(df.SupportsTreeBulkIO<int>("z").first);
   EXPECT_TRUE(df.SupportsTreeBulkIO<ROOT::RVec<float>>("arr").first);

   EXPECT_THROW(df.SupportsTreeBulkIO<double>("arr"), std::runtime_error);
   EXPECT_THROW(df.SupportsTreeBulkIO<double>("x"), std::runtime_error);
}

// the type of a column defined via the bulk API must be inferred from
// the second argument to the callable rather than from its return type
TEST(RDFBulkAPI, DefinedColumnType)
{
   auto df = ROOT::RDataFrame(1);
   auto bulkX = [](const REventMask &, ROOT::RVecI &) {};
   auto df2 = df.Define("x", bulkX);
   EXPECT_EQ(df2.GetColumnType("x"), "int");
}

/**************** These tests are sequential and IMT (defined with TEST_P) *********************/
class RDFBulkAPI : public ::testing::TestWithParam<bool> {
protected:
   RDFBulkAPI() : NSLOTS(GetParam() ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam())
         ROOT::EnableImplicitMT(NSLOTS);
   }

   ~RDFBulkAPI()
   {
      if (GetParam())
         ROOT::DisableImplicitMT();
   }

   const unsigned int NSLOTS;
};

TEST_P(RDFBulkAPI, BulkDefine)
{
   auto bulkReturnX = [](const REventMask &m, ROOT::RVec<float> &output, const ROOT::RVec<ULong64_t> &rdfentries) {
      // ignoring event mask
      std::copy(rdfentries.begin(), rdfentries.begin() + m.Size(), output.begin());
   };

   auto bulkSqr = [](const REventMask &m, ROOT::RVec<float> &output, const ROOT::RVec<float> &xs) {
      // ignoring event mask
      std::transform(xs.begin(), xs.begin() + m.Size(), output.begin(), [](float x) { return x * x; });
   };

   auto m = ROOT::RDataFrame(10).Define("x", bulkReturnX, {"rdfentry_"}).Define("xx", bulkSqr, {"x"}).Mean<float>("xx");
   EXPECT_DOUBLE_EQ(m.GetValue(), 28.5);
}

struct BulkHelper : ROOT::Detail::RDF::RActionImpl<BulkHelper> {
   static constexpr bool kUseBulk = true;
   using Result_t = int;
   std::shared_ptr<int> x = std::make_shared<int>(42);

   std::shared_ptr<int> GetResultPtr() const { return x; }

   void Exec(unsigned int slot, const REventMask &m, const ROOT::RVecULL &es)
   {
      EXPECT_EQ(m.FirstEntry(), 0ull);
      EXPECT_EQ(m.Size(), 10ull);
      for (ULong64_t i = 0ull; i < 10ull; ++i)
         EXPECT_EQ(i, es[i]);
   }

   void Initialize() {}
   void InitTask(TTreeReader *, unsigned) {}
   void Finalize() {}
   std::string GetActionName() { return "custom"; }
};

// TODO: change to TEST_P when we figure out what condition to check on for MT runs
TEST(RDFBulkAPI, BulkAction)
{
   auto df = ROOT::RDataFrame(10);
   auto r = df.Book<ULong64_t>(BulkHelper{}, {"rdfentry_"});
   r.GetValue();
}

/*
TEST(RDFBulkAPI, BulkFilter)
{
   auto bulkReturnX = [](const ROOT::RDF::Experimental::REventMask &m, ROOT::RVec<float> &output,
                         const ROOT::RVec<ULong64_t> &rdfentries) {
      // ignoring event mask
      std::copy(rdfentries.begin(), rdfentries.begin() + m.Size(), output.begin());
   };

   auto bulkSqr = [](const ROOT::RDF::Experimental::REventMask &m, std::size_t bulkSize, ROOT::RVec<float> &output,
                     const ROOT::RVec<float> &xs) {
      for (std::size_t i = 0u; i < m.Size(); ++i) {
         if (m[i]) {
            output[i] = xs[i] * xs[i];
         }
      }
   };

   auto m =
      ROOT::RDataFrame(10)
         .Filter([](const ROOT::RDF::Experimental::REventMask &, std::size_t bulkSize, ROOT::RVec<bool> &outputMask,
                    const ROOT::RVec<ULong64_t> &rdfentries) { outputMask = rdfentries > 5; },
                 {"rdfentry_"})
         .Define("x", bulkReturnX, {"rdfentry_"})
         .Define("xx", bulkSqr, {"x"})
         .Mean<double>("xx");
   EXPECT_DOUBLE_EQ(m.GetValue(), 57.5);
}
*/

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDFBulkAPI, ::testing::Values(false));

// instantiate multi-thread tests
#ifdef R__USE_IMT
INSTANTIATE_TEST_SUITE_P(MT, RDFBulkAPI, ::testing::Values(true));
#endif
