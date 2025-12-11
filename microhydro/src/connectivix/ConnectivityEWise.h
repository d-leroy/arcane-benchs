#ifndef CONNECTIVIX_CONNECTIVITY_MAT_MUL_H
#define CONNECTIVIX_CONNECTIVITY_MAT_MUL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "CSR.h"
#include "Metadata.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/Atomic.h"
#include "arcane/accelerator/LocalMemory.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLaunch.h"
#include "define.h"
#include <cstddef>
#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;
using namespace Arcane;

namespace ax = Arcane::Accelerator;

namespace Connectivix {

class ConnectivityEWiseMult {
public:
  //

  template <typename IndexType, typename AllocType> struct SpMatrixEWiseMult {
    template <typename T> using ContainerType = thrust::device_vector<T, typename AllocType::template rebind<T>::other>;
    using MatrixType = nsparse::matrix<bool, IndexType, AllocType>;
    using Int32 = unsigned long;

    static_assert(sizeof(Int32) > sizeof(IndexType), "Values intersection index must be larger");

    static void fillIndices(const MatrixType &m, ContainerType<Int32> &out) {
      thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(m.m_vals),
                       [rowOffset = m.m_row_index.data(), colIndex = m.m_col_index.data(), outIndices = out.data(), nrows = m.m_rows, ncols = m.m_cols] __device__(IndexType valueId) {
                         Int32 row = findNearestRowIdx<index>(valueId, nrows, rowOffset);
                         Int32 col = colIndex[valueId];
                         Int32 index = row * ncols + col;
                         outIndices[valueId] = index;
                       });
    }

    MatrixType operator()(const MatrixType &a, const MatrixType &b) {
      auto aNvals = a.m_vals;
      auto bNvals = b.m_vals;
      auto worst = std::min(aNvals, bNvals);

      // Allocate memory for the worst case scenario
      NumArray<Int32, MDDim1> inputA(aNvals);
      NumArray<Int32, MDDim1> inputB(bNvals);

      fillIndices(a, inputA);
      fillIndices(b, inputB);

      NumArray<Int32, MDDim1> intersected(worst);

      auto out = thrust::set_intersection(inputA.begin(), inputA.end(), inputB.begin(), inputB.end(), intersected.begin());

      // Count result nvals count
      auto nvals = thrust::distance(intersected.begin(), out);

      NumArray<Int32, MDDim1> rowOffsetTmp(a.m_rows + 1);
      NumArray<Int32, MDDim1> colIndex(nvals);

      thrust::fill(rowOffsetTmp.begin(), rowOffsetTmp.end(), 0);

      thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(nvals),
                       [rowOffset = rowOffsetTmp.data(), colIndex = colIndex.data(), intersected = intersected.data(), nrows = a.m_rows, ncols = a.m_cols] __device__(IndexType valueId) {
                         Int32 i = intersected[valueId];
                         Int32 row = i / ncols;
                         Int32 col = i % ncols;
                         atomicAdd((rowOffset + row).get(), 1);
                         colIndex[valueId] = (IndexType)col;
                       });

      NumArray<Int32, MDDim1> rowOffset(a.m_rows + 1);
      thrust::exclusive_scan(rowOffsetTmp.begin(), rowOffsetTmp.end(), rowOffset.begin(), 0, thrust::plus<index>());

      assert(nvals == rowOffset.back());

      return MatrixType(std::move(colIndex), std::move(rowOffset), a.m_rows, a.m_cols, nvals);
    }
  };
};
} // namespace Connectivix