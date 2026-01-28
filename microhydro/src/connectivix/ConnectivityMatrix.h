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
#include "define.h"
#include <string>
#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;
using namespace Arcane;
namespace ax = Arcane::Accelerator;

namespace Connectivix {

template <typename ItemType1, typename ItemType2> class ConnectivityMatrix {
public:
  using ItemType1Type = ItemType1;
  using ItemType2Type = ItemType2;
  using ItemLocalId1 = typename ItemType1::LocalIdType;
  using ItemLocalId2 = typename ItemType2::LocalIdType;

  ConnectivityMatrix(Int32 M, Int32 N) {
    m_data = new CSR(M, N);
  };
  ~ConnectivityMatrix() {
    delete m_data;
  };

  void build(const IndexedItemConnectivityGenericViewT<ItemType1, ItemType2> from, const ItemGroupT<ItemType1> items) {
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

  // void clone(const ConnectivityMatrix<ItemType1, ItemType2> &otherBase);
  // ConnectivityMatrix<ItemType2, ItemType1> transpose(bool checkTime);

  template <typename ItemType3> ConnectivityMatrix<ItemType1, ItemType3> *matMul(const ConnectivityMatrix<ItemType2, ItemType3> &bBase, ax::Runner &runner) {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<ItemType1, ItemType3> *result = new ConnectivityMatrix<ItemType1, ItemType3>(cRows, cCols);
    ConnectivityMatMul matMul(*m_data, *bBase.m_data, *result->m_data, runner);
    matMul.doMatMul();

    return result;
  }

  ConnectivityMatrix<ItemType2, ItemType1> *transpose(ax::Runner &runner) {
    const Int32 tRows = getNbCols();
    const Int32 tCols = getNbRows();

    ConnectivityMatrix<ItemType2, ItemType1> *result = new ConnectivityMatrix<ItemType2, ItemType1>(tRows, tCols);
    result->m_data = m_data->transpose();
    // ConnectivityTranspose transpose(*m_data, *result->m_data, runner);
    // transpose.doTranspose();

    return result;
  }

  ConnectivityMatrix<ItemType1, ItemType2> *eWiseMatMul(const ConnectivityMatrix<ItemType1, ItemType2> &bBase, ax::Runner &runner) {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<ItemType1, ItemType2> *result = new ConnectivityMatrix<ItemType1, ItemType2>(cRows, cCols);
    ConnectivityEWiseMatMul eWiseMatMul(*m_data, *bBase.m_data, *result->m_data, runner);
    eWiseMatMul.doEWiseMatMul();

    return result;
  }

  ConnectivityMatrix<ItemType1, ItemType2> *eWiseMatSub(const ConnectivityMatrix<ItemType1, ItemType2> &bBase, ax::Runner &runner) {
    const Int32 cRows = getNbRows();
    const Int32 cCols = bBase.getNbCols();

    ConnectivityMatrix<ItemType1, ItemType2> *result = new ConnectivityMatrix<ItemType1, ItemType2>(cRows, cCols);
    ConnectivityEWiseMatSub eWiseMatSub(*m_data, *bBase.m_data, *result->m_data, runner);
    eWiseMatSub.doEWiseMatSub();

    return result;
  }

  // ConnectivityMatrix<ItemType1, ItemType2>
  // vecMul(const ItemVectorBase<ItemType2> &bBaseDiag, bool checkTime);
  // TODO: identical to product with single column matrix ?

  // ConnectivityMatrix<ItemType1, ItemType2>
  // eWiseAdd(const ConnectivityMatrix<ItemType1, ItemType2> &bBase,
  //          bool checkTime);

  // const ItemVectorBase<ItemType2> rowVector(ItemLocalId1 i);

  // ARCCORE_HOST_DEVICE const ItemIterator<ItemType2> row(ItemLocalId1 i)
  // const
  // {
  //   return ItemIterator<ItemType2>{

  //   };
  // };

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

} // namespace Connectivix

#endif // CONNECTIVIX_CONNECTIVITY_MATRIX_H