

/**
 \file ROOT/RDF/SYCLFillHelper.hxx
 \ingroup dataframe
 \author Jolly Chen, CERN
 \date 2023-06
*/

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_SYCLFILLHELPER
#define ROOT_SYCLFILLHELPER

#include <ROOT/RDF/RAction.hxx>
#include "ROOT/RDF/RActionImpl.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "RHnSYCL.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TStatistic.h"

#include <vector>
#include <string>
#include <array>

using ROOT::Internal::RDF::Disjunction;
using ROOT::Internal::RDF::GetNthElement;
using ROOT::Internal::RDF::IsDataContainer;
using ROOT::Internal::RDF::RActionImpl;
using ROOT::Internal::RDF::RMergeableFill;
using ROOT::Internal::RDF::RMergeableValueBase;

namespace ROOT {
namespace Experimental {
using Hist_t = ::TH1D;

template <typename HIST = Hist_t>
class R__CLING_PTRCHECK(off) SYCLFillHelper : public RActionImpl<SYCLFillHelper<HIST>> {
   // clang-format off
   static constexpr size_t getHistDim(TH3 *) { return 3; }
   static constexpr size_t getHistDim(TH2 *) { return 2; }
   static constexpr size_t getHistDim(TH1 *) { return 1; }

   static constexpr char getHistType(TH1C *) { return (char) 0; }
   static constexpr char getHistType(TH2C *) { return (char) 0; }
   static constexpr char getHistType(TH3C *) { return (char) 0; }
   static constexpr short getHistType(TH1S *) { return (short) 0; }
   static constexpr short getHistType(TH2S *) { return (short) 0; }
   static constexpr short getHistType(TH3S *) { return (short) 0; }
   static constexpr int getHistType(TH1I *) { return 0; }
   static constexpr int getHistType(TH2I *) { return 0; }
   static constexpr int getHistType(TH3I *) { return 0; }
   static constexpr float getHistType(TH1F *) { return (float) 0; }
   static constexpr float getHistType(TH2F *) { return (float) 0; }
   static constexpr float getHistType(TH3F *) { return (float) 0; }
   static constexpr double getHistType(TH1D *) { return (double) 0; }
   static constexpr double getHistType(TH2D *) { return (double) 0; }
   static constexpr double getHistType(TH3D *) { return (double) 0; }
   // clang-format on

   static constexpr size_t dim = getHistDim((HIST *)nullptr);
   using SYCLHist_t = ROOT::Experimental::RHnSYCL<decltype(getHistType((HIST *)nullptr)), dim>;

   HIST *fObject;
   std::unique_ptr<SYCLHist_t> fSYCLHist;

   template <typename H = HIST, typename = decltype(std::declval<H>().Reset())>
   void ResetIfPossible(H *h)
   {
      h->Reset();
   }

   void ResetIfPossible(TStatistic *h) { *h = TStatistic(); }

   // cannot safely re-initialize variations of the result, hence error out
   void ResetIfPossible(...)
   {
      throw std::runtime_error(
         "A systematic variation was requested for a custom Fill action, but the type of the object to be filled does"
         "not implement a Reset method, so we cannot safely re-initialize variations of the result. Aborting.");
   }

   void UnsetDirectoryIfPossible(TH1 *h) { h->SetDirectory(nullptr); }

   void UnsetDirectoryIfPossible(...) {}

   template <size_t DIMW>
   void FillWithWeight(unsigned int slot, const std::array<double, DIMW> &v)
   {
      double w = v.back();
      std::array<double, DIMW - 1> coords;
      std::copy(v.begin(), v.end() - 1, coords.begin());
      fSYCLHist->Fill(coords, w);
   }

   template <typename... Coords>
   void FillWithoutWeight(unsigned int slot, const Coords &...x)
   {
      fSYCLHist->Fill({x...});
   }

   // Merge overload for types with Merge(TCollection*), like TH1s
   template <typename H, typename = std::enable_if_t<std::is_base_of<TObject, H>::value, int>>
   auto Merge(std::vector<H *> &objs, int /*toincreaseoverloadpriority*/)
      -> decltype(objs[0]->Merge((TCollection *)nullptr), void())
   {
      TList l;
      for (auto it = ++objs.begin(); it != objs.end(); ++it)
         l.Add(*it);
      objs[0]->Merge(&l);
   }

   // Merge overload for types with Merge(const std::vector&)
   template <typename H>
   auto Merge(std::vector<H *> &objs, double /*toloweroverloadpriority*/)
      -> decltype(objs[0]->Merge(std::vector<HIST *>{}), void())
   {
      objs[0]->Merge({++objs.begin(), objs.end()});
   }

   // Merge overload to error out in case no valid HIST::Merge method was detected
   template <typename T>
   void Merge(T, ...)
   {
      static_assert(
         sizeof(T) < 0,
         "The type passed to Fill does not provide a Merge(TCollection*) or Merge(const std::vector&) method.");
   }

   // class which wraps a pointer and implements a no-op increment operator
   template <typename T>
   class ScalarConstIterator {
      const T *obj_;

   public:
      ScalarConstIterator(const T *obj) : obj_(obj) {}
      const T &operator*() const { return *obj_; }
      ScalarConstIterator<T> &operator++() { return *this; }
   };

   // helper functions which provide one implementation for scalar types and another for containers
   // TODO these could probably all be replaced by inlined lambdas and/or constexpr if statements
   // in c++17 or later

   // return unchanged value for scalar
   template <typename T, std::enable_if_t<!IsDataContainer<T>::value, int> = 0>
   ScalarConstIterator<T> MakeBegin(const T &val)
   {
      return ScalarConstIterator<T>(&val);
   }

   // return iterator to beginning of container
   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   auto MakeBegin(const T &val)
   {
      return std::begin(val);
   }

   // return 1 for scalars
   template <typename T, std::enable_if_t<!IsDataContainer<T>::value, int> = 0>
   std::size_t GetSize(const T &)
   {
      return 1;
   }

   // return container size
   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   std::size_t GetSize(const T &val)
   {
#if __cplusplus >= 201703L
      return std::size(val);
#else
      return val.size();
#endif
   }

   template <std::size_t ColIdx, typename End_t, typename... Its>
   void ExecLoop(unsigned int slot, End_t end, Its... its)
   {
      auto *thisSlotH = fObject;
      // loop increments all of the iterators while leaving scalars unmodified
      // TODO this could be simplified with fold expressions or std::apply in C++17
      auto nop = [](auto &&...) {};
      for (; GetNthElement<ColIdx>(its...) != end; nop(++its...)) {
         thisSlotH->Fill(*its...);
      }
   }

public:
   SYCLFillHelper(SYCLFillHelper &&) = default;
   SYCLFillHelper(const SYCLFillHelper &) = delete;

   // Initialize fSYCLHist
   inline void init_sycl(HIST *obj, int i)
   {
      if (getenv("DBG"))
         printf("Init sycl hist %d\n", i);
      auto dims = obj->GetDimension();
      std::array<Int_t, dim> ncells;
      std::array<Double_t, dim> xlow;
      std::array<Double_t, dim> xhigh;
      std::array<const Double_t *, dim> binEdges;
      TAxis *ax;

      for (auto d = 0; d < dims; d++) {
         if (d == 0) {
            ax = obj->GetXaxis();
         } else if (d == 1) {
            ax = obj->GetYaxis();
         } else {
            ax = obj->GetZaxis();
         }

         ncells[d] = ax->GetNbins() + 2;
         binEdges[d] = ax->GetXbins()->GetArray();
         xlow[d] = ax->GetXmin();
         xhigh[d] = ax->GetXmax();

         if (getenv("DBG"))
            printf("\tdim %d --- nbins: %d xlow: %f xhigh: %f\n", d, ncells[d], xlow[d], xhigh[d]);
      }

      fSYCLHist = std::make_unique<SYCLHist_t>(ncells, xlow, xhigh, binEdges.data());
   }

   SYCLFillHelper(const std::shared_ptr<HIST> &h, const unsigned int nSlots)
   {
      // We ignore nSlots and just create one SYCLHist instance that handles the parallelization.
      fObject = h.get();
      init_sycl(fObject, 0);
   }

   void InitTask(TTreeReader *, unsigned int) {}

   template <typename... ValTypes, std::enable_if_t<!Disjunction<IsDataContainer<ValTypes>...>::value, int> = 0>
   auto Exec(unsigned int slot, const ValTypes &...x)
   {
      if constexpr (sizeof...(ValTypes) > dim)
         FillWithWeight<dim + 1>(slot, {((double)x)...});
      else
         FillWithoutWeight(slot, x...);
      return;

      fObject->Fill(x...);
   }

   // at least one container argument
   template <typename... Xs, std::enable_if_t<Disjunction<IsDataContainer<Xs>...>::value, int> = 0>
   auto Exec(unsigned int slot, const Xs &...xs) -> decltype(fObject->Fill(*MakeBegin(xs)...), void())
   {
      // array of bools keeping track of which inputs are containers
      constexpr std::array<bool, sizeof...(Xs)> isContainer{IsDataContainer<Xs>::value...};

      // index of the first container input
      constexpr std::size_t colidx = FindIdxTrue(isContainer);
      // if this happens, there is a bug in the implementation
      static_assert(colidx < sizeof...(Xs), "Error: index of collection-type argument not found.");

      // get the end iterator to the first container
      auto const xrefend = std::end(GetNthElement<colidx>(xs...));

      // array of container sizes (1 for scalars)
      std::array<std::size_t, sizeof...(xs)> sizes = {{GetSize(xs)...}};

      for (std::size_t i = 0; i < sizeof...(xs); ++i) {
         if (isContainer[i] && sizes[i] != sizes[colidx]) {
            throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
         }
      }

      ExecLoop<colidx>(slot, xrefend, MakeBegin(xs)...);
   }

   template <typename T = HIST>
   void Exec(...)
   {
      static_assert(sizeof(T) < 0,
                    "When filling an object with RDataFrame (e.g. via a Fill action) the number or types of the "
                    "columns passed did not match the signature of the object's `Fill` method.");
   }

   void Initialize()
   { /* noop */
   }

   void Finalize()
   {
      double stats[13];

      HIST *h = fObject;
      fSYCLHist->RetrieveResults(h->GetArray(), stats);
      h->SetStatsData(stats);
      h->SetEntries(fSYCLHist->GetEntries());
      // printf("%d %d??\n", fObjects[i]->GetArray()->size(), fObjects[i]->GetXaxis()->GetNbins());
      if (getenv("DBG")) {
         printf("sycl stats:");
         for (int j = 0; j < 13; j++) {
            printf("%f ", stats[j]);
         }
         printf(" %f\n", fObject->GetEntries());
      }

      if (getenv("DBG")) {
         fObject->GetStats(stats);
         printf("stats:");
         for (int j = 0; j < 13; j++) {
            printf("%f ", stats[j]);
         }
         printf(" %f\n", fObject->GetEntries());
         if (getenv("DBG") && atoi(getenv("DBG")) > 1) {

            printf("histogram:");
            for (int j = 0; j < fObject->GetNcells(); ++j) {
               printf("%f ", fObject->GetArray()[j]);
            }
            printf("\n");
         }
      }
   }

   HIST &PartialUpdate(unsigned int slot)
   {
      return *fObject;
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<HIST>>(*fObject);
   }

   // if the fObjects vector type is derived from TObject, return the name of the object
   template <typename T = HIST, std::enable_if_t<std::is_base_of<TObject, T>::value, int> = 0>
   std::string GetActionName()
   {
      return std::string(fObject->IsA()->GetName()) + "\\n" + std::string(fObject->GetName());
   }

   // if fObjects is not derived from TObject, indicate it is some other object
   template <typename T = HIST, std::enable_if_t<!std::is_base_of<TObject, T>::value, int> = 0>
   std::string GetActionName()
   {
      return "Fill SYCL histogram";
   }

   template <typename H = HIST>
   SYCLFillHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<H> *>(newResult);
      ResetIfPossible(result.get());
      UnsetDirectoryIfPossible(result.get());
      return SYCLFillHelper(result, 1);
   }
};

} // namespace Experimental
} // namespace ROOT
#endif
