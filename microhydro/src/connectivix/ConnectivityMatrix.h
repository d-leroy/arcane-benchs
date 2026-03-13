#ifndef CONNECTIVIX_CONNECTIVITY_MATRIX_H
#define CONNECTIVIX_CONNECTIVITY_MATRIX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "CSR.h"
#include "ConnectivityMatMul.h"
#include "Metadata.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/Atomic.h"
#include "arcane/accelerator/LocalMemory.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLaunch.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/core/IndexedItemConnectivityView.h"
#include "arcane/mesh/ItemFamily.h"
#include "arcane/utils/MemoryUtils.h"
#include "connectivix/ConnectivityEWiseMatMul.h"
#include "connectivix/ConnectivityEWiseMatSub.h"
#include "connectivix/ConnectivityTranspose.h"
#include "connectivix/ConnectivityVector.h"
#include "define.h"
#include <concepts>
#include <string>
#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;
using namespace Arcane;
namespace ax = Arcane::Accelerator;

namespace Connectivix {

template <typename T>
concept ConnectivityMatrixViewC = requires(const T m, const T::RowType i, const T::ColType j) {
  typename T::RowType;
  typename T::ColType;
  { m.row(i) } -> ConnectivityVectorC;
  { m.indexInRow(i, j) } -> std::convertible_to<Int32>;

  requires std::same_as<typename decltype(m.row(i))::ItemType, typename T::ColType>;
};

template <typename T>
concept ConnectivityMatrixC = requires(const T m, ax::RunCommand &command) {
  typename T::RowType;
  typename T::ColType;
  { m.m_data } -> std::convertible_to<Connectivix::CSR *>;
  { m.view(command) } -> ConnectivityMatrixViewC;

  requires std::same_as<typename decltype(m.view(command))::RowType, typename T::RowType::LocalIdType>;
  requires std::same_as<typename decltype(m.view(command))::ColType, typename T::ColType::LocalIdType>;
};

// template <typename L, typename R>
// concept Multiplicable = ConnectivityMatrixC<L> && ConnectivityMatrixC<R> && std::same_as<typename L::ColType, typename R::RowType> && requires(L l, R r) {
//   { l.matMul(r) } -> ConnectivityMatrixC;
// };

template <typename ItemLocalId1, typename ItemLocalId2> class ConnectivityMatrixView {
public:
  using RowType = ItemLocalId1;
  using ColType = ItemLocalId2;

public:
  ConnectivityMatrixView(const Int32 M, const Int32 N, const ax::NumArrayInView<Int32, MDDim1> &in_rpt, const ax::NumArrayInView<Int32, MDDim1> &in_col) : M(M), N(N), in_rpt(in_rpt), in_col(in_col) {}

private:
  const Int32 M;
  const Int32 N;
  const ax::NumArrayInView<Int32, MDDim1> in_rpt;
  const ax::NumArrayInView<Int32, MDDim1> in_col;

public:
  ARCCORE_HOST_DEVICE inline const ConnectivityVector<ColType> row(RowType item) const {
    auto from = in_rpt[item];
    auto size = in_rpt[item + 1] - from;
    return ConnectivityVector<ColType>(in_col.to1DSpan().subSpan(from, size));
  }

  ARCCORE_HOST_DEVICE inline Int32 indexInRow(RowType row, ColType col) const {
    auto from = in_rpt[row];
    auto to = in_rpt[row + 1] - from;
    Int32 result = 0;
    for (Int32 k = 0; k < to; ++k) {
      result += (unsigned int)(in_col[from + k] - col) >> 31;
    }
    Int32 found = (unsigned int)(result - to) >> 31;
    return result * found + (1 - found) * -1;
  }
};

template <typename ItemLocalId1, typename ItemLocalId2> class OrderedConnectivityMatrixView {
public:
  using RowType = ItemLocalId1;
  using ColType = ItemLocalId2;

public:
  OrderedConnectivityMatrixView(const Int32 M, const Int32 N, const ax::NumArrayInView<Int32, MDDim1> &in_rpt, const ax::NumArrayInView<Int32, MDDim1> &in_col,
                                const ax::NumArrayInView<Int32, MDDim1> &in_order)
      : M(M), N(N), in_rpt(in_rpt), in_col(in_col), in_order(in_order) {}

private:
  const Int32 M;
  const Int32 N;
  const ax::NumArrayInView<Int32, MDDim1> in_rpt;
  const ax::NumArrayInView<Int32, MDDim1> in_col;
  const ax::NumArrayInView<Int32, MDDim1> in_order;

public:
  ARCCORE_HOST_DEVICE inline const OrderedConnectivityVector<ColType> row(RowType item) const {
    auto from = in_rpt[item];
    auto size = in_rpt[item + 1] - from;
    return OrderedConnectivityVector<ColType>(in_col.to1DSpan().subSpan(from, size), in_order.to1DSpan().subSpan(from, size));
  }

  ARCCORE_HOST_DEVICE inline Int32 indexInRow(RowType row, ColType col) const {
    auto from = in_rpt[row];
    auto to = in_rpt[row + 1] - from;
    Int32 result = 0;
    for (Int32 k = 0; k < to; ++k) {
      result += (unsigned int)(in_col[from + k] - col) >> 31;
    }
    Int32 found = (unsigned int)(result - to) >> 31;
    return in_order[result * found + (1 - found) * -1];
  }
};

template <typename ItemType1, typename ItemType2> class ConnectivityMatrix {
public:
  using RowType = ItemType1;
  using ColType = ItemType2;
  using RowLocalIdType = typename RowType::LocalIdType;
  using ColLocalIdType = typename ColType::LocalIdType;

  ConnectivityMatrix(Int32 M, Int32 N) {
    m_data = new CSR(M, N);
  };
  ~ConnectivityMatrix() {
    delete m_data;
  };

  void build(const IndexedItemConnectivityGenericViewT<RowType, ColType> from, const ItemGroupT<RowType> items) {
    std::vector<Int32> I_vector, J_vector;
    ENUMERATE_ITEM(iitem, items) {
      const auto sourceItemId = *iitem;
      for (const auto connectedItem : from.items(iitem)) {
        I_vector.push_back(sourceItemId.localId());
        J_vector.push_back(connectedItem.localId());
      }
    }
    Int32 nnz = J_vector.size();

    m_data->fromCoordinates(I_vector.data(), J_vector.data(), nnz);
  }

  template <typename Other>
    requires std::same_as<ColType, typename Other::RowType> && ConnectivityMatrixC<Other>
  auto matMul(const Other &bBase, ax::Runner &runner) const {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<RowType, typename Other::ColType> *result = new ConnectivityMatrix<RowType, typename Other::ColType>(cRows, cCols);
    ConnectivityMatMul matMul(*m_data, *bBase.m_data, *result->m_data, runner);
    matMul.doMatMul();

    return result;
  }

  ConnectivityMatrix<ColType, RowType> *transpose(ax::Runner &runner) const {
    const Int32 tRows = getNbCols();
    const Int32 tCols = getNbRows();

    ConnectivityMatrix<ColType, RowType> *result = new ConnectivityMatrix<ColType, RowType>(tRows, tCols);
    result->m_data = m_data->transpose();

    return result;
  }

  template <typename Other>
    requires std::same_as<RowType, typename Other::RowType> && std::same_as<ColType, typename Other::ColType> && ConnectivityMatrixC<Other>
  ConnectivityMatrix<RowType, ColType> *eWiseMatMul(const Other &bBase, ax::Runner &runner) const {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<RowType, ColType> *result = new ConnectivityMatrix<RowType, ColType>(cRows, cCols);
    ConnectivityEWiseMatMul eWiseMatMul(*m_data, *bBase.m_data, *result->m_data, runner);
    eWiseMatMul.doEWiseMatMul();

    return result;
  }

  template <typename Other>
    requires std::same_as<RowType, typename Other::RowType> && std::same_as<ColType, typename Other::ColType> && ConnectivityMatrixC<Other>
  ConnectivityMatrix<RowType, ColType> *eWiseMatSub(const Other &bBase, ax::Runner &runner) const {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<RowType, ColType> *result = new ConnectivityMatrix<RowType, ColType>(cRows, cCols);
    ConnectivityEWiseMatSub eWiseMatSub(*m_data, *bBase.m_data, *result->m_data, runner);
    eWiseMatSub.doEWiseMatSub();

    return result;
  }

  template <typename Other>
    requires std::same_as<RowType, typename Other::RowType> && std::same_as<ColType, typename Other::ColType> && ConnectivityMatrixC<Other>
  ConnectivityMatrix<RowType, ColType> *intersect(const Other &bBase, ax::Runner &runner) const {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<RowType, ColType> *result = new ConnectivityMatrix<RowType, ColType>(cRows, cCols);
    ConnectivityEWiseMatMul eWiseMatMul(*m_data, *bBase.m_data, *result->m_data, runner);
    eWiseMatMul.doEWiseMatMul();

    return result;
  }

  template <typename Other>
    requires std::same_as<RowType, typename Other::RowType> && std::same_as<ColType, typename Other::ColType> && ConnectivityMatrixC<Other>
  ConnectivityMatrix<RowType, ColType> *subtract(const Other &bBase, ax::Runner &runner) const {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<RowType, ColType> *result = new ConnectivityMatrix<RowType, ColType>(cRows, cCols);
    ConnectivityEWiseMatSub eWiseMatSub(*m_data, *bBase.m_data, *result->m_data, runner);
    eWiseMatSub.doEWiseMatSub();

    return result;
  }

  ConnectivityMatrixView<RowLocalIdType, ColLocalIdType> view(ax::RunCommand &command) const {
    auto rpt_view = ax::viewIn(command, *m_data->rpt);
    auto col_view = ax::viewIn(command, *m_data->col);
    auto result = ConnectivityMatrixView<RowLocalIdType, ColLocalIdType>(m_data->M, m_data->N, rpt_view, col_view);

    return result;
  }

  std::string printStorage() const {
    return m_data->printRows() + '\n' + m_data->printCols();
  }

  Int32 getNbRows() const {
    return m_data->M;
  };
  Int32 getNbCols() const {
    return m_data->N;
  };
  Int32 getNbVals() const {
    return m_data->nnz;
  };

public:
  Connectivix::CSR *m_data;
};

template <typename ItemType1, typename ItemType2> class OrderedConnectivityMatrix {
public:
  using RowType = ItemType1;
  using ColType = ItemType2;
  using RowLocalIdType = typename RowType::LocalIdType;
  using ColLocalIdType = typename ColType::LocalIdType;

  OrderedConnectivityMatrix(Int32 M, Int32 N) {
    m_data = new CSR(M, N);
  };
  ~OrderedConnectivityMatrix() {
    delete m_data;
    delete m_order;
  };

  void build(const IndexedItemConnectivityGenericViewT<RowType, ColType> from, const ItemGroupT<RowType> items) {
    std::vector<Int32> I_vector, J_vector, order_vector;
    ENUMERATE_ITEM(iitem, items) {
      const auto sourceItemId = *iitem;
      Int32 idx = 0;
      for (const auto connectedItem : from.items(iitem)) {
        I_vector.push_back(sourceItemId.localId());
        J_vector.push_back(connectedItem.localId());
        order_vector.push_back(idx);
        ++idx;
      }
    }
    Int32 nnz = J_vector.size();

    m_data->fromCoordinatesOrdered(I_vector.data(), J_vector.data(), nnz, order_vector.data());

    m_order = new NumArray<Int32, MDDim1>(nnz, m_data->col->memoryResource());
    Span<const Int32> spanOrder(order_vector.data(), order_vector.size());
    m_order->copy(spanOrder);
  }

  template <typename Other>
    requires std::same_as<ColType, typename Other::RowType> && ConnectivityMatrixC<Other>
  auto matMul(const Other &bBase, ax::Runner &runner) {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<RowType, typename Other::ColType> *result = new ConnectivityMatrix<RowType, typename Other::ColType>(cRows, cCols);
    ConnectivityMatMul matMul(*m_data, *bBase.m_data, *result->m_data, runner);
    matMul.doMatMul();

    return result;
  }

  ConnectivityMatrix<ColType, RowType> *transpose(ax::Runner &runner) {
    const Int32 tRows = getNbCols();
    const Int32 tCols = getNbRows();

    ConnectivityMatrix<ColType, RowType> *result = new ConnectivityMatrix<ColType, RowType>(tRows, tCols);
    result->m_data = m_data->transpose();

    return result;
  }

  template <typename Other>
    requires std::same_as<RowType, typename Other::RowType> && std::same_as<ColType, typename Other::ColType> && ConnectivityMatrixC<Other>
  ConnectivityMatrix<RowType, ColType> *eWiseMatMul(const Other &bBase, ax::Runner &runner) {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<RowType, ColType> *result = new ConnectivityMatrix<RowType, ColType>(cRows, cCols);
    ConnectivityEWiseMatMul eWiseMatMul(*m_data, *bBase.m_data, *result->m_data, runner);
    eWiseMatMul.doEWiseMatMul();

    return result;
  }

  template <typename Other>
    requires std::same_as<RowType, typename Other::RowType> && std::same_as<ColType, typename Other::ColType> && ConnectivityMatrixC<Other>
  ConnectivityMatrix<RowType, ColType> *eWiseMatSub(const Other &bBase, ax::Runner &runner) {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<RowType, ColType> *result = new ConnectivityMatrix<RowType, ColType>(cRows, cCols);
    ConnectivityEWiseMatSub eWiseMatSub(*m_data, *bBase.m_data, *result->m_data, runner);
    eWiseMatSub.doEWiseMatSub();

    return result;
  }

  template <typename Other>
    requires std::same_as<RowType, typename Other::RowType> && std::same_as<ColType, typename Other::ColType> && ConnectivityMatrixC<Other>
  ConnectivityMatrix<RowType, ColType> *intersect(const Other &bBase, ax::Runner &runner) {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<RowType, ColType> *result = new ConnectivityMatrix<RowType, ColType>(cRows, cCols);
    ConnectivityEWiseMatMul eWiseMatMul(*m_data, *bBase.m_data, *result->m_data, runner);
    eWiseMatMul.doEWiseMatMul();

    return result;
  }

  template <typename Other>
    requires std::same_as<RowType, typename Other::RowType> && std::same_as<ColType, typename Other::ColType> && ConnectivityMatrixC<Other>
  ConnectivityMatrix<RowType, ColType> *subtract(const Other &bBase, ax::Runner &runner) {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<RowType, ColType> *result = new ConnectivityMatrix<RowType, ColType>(cRows, cCols);
    ConnectivityEWiseMatSub eWiseMatSub(*m_data, *bBase.m_data, *result->m_data, runner);
    eWiseMatSub.doEWiseMatSub();

    return result;
  }

  OrderedConnectivityMatrixView<RowLocalIdType, ColLocalIdType> view(ax::RunCommand &command) const {
    auto rpt_view = ax::viewIn(command, *m_data->rpt);
    auto col_view = ax::viewIn(command, *m_data->col);
    auto order_view = ax::viewIn(command, *m_order);
    auto result = OrderedConnectivityMatrixView<RowLocalIdType, ColLocalIdType>(m_data->M, m_data->N, rpt_view, col_view, order_view);

    return result;
  }

  std::string printStorage() const {
    return m_data->printRows() + '\n' + m_data->printCols();
  }

  Int32 getNbRows() const {
    return m_data->M;
  };
  Int32 getNbCols() const {
    return m_data->N;
  };
  Int32 getNbVals() const {
    return m_data->nnz;
  };

  // FIXME: can this be private?
public:
  Connectivix::CSR *m_data;

private:
  NumArray<Int32, MDDim1> *m_order;
};

} // namespace Connectivix

#endif // CONNECTIVIX_CONNECTIVITY_MATRIX_H