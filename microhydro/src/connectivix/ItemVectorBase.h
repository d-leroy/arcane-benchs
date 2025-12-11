#ifndef CONNECTIVIX_ITEM_VECTOR_BASE_H
#define CONNECTIVIX_ITEM_VECTOR_BASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Connectivix {

template <typename ItemType> class ItemVectorBase {
public:
  using ItemTypeType = ItemType;
  using ItemLocalId = typename ItemType::LocalIdType;

  ItemVectorBase() = default;
  ~ItemVectorBase() = default;
};
} // namespace Connectivix

#endif // CONNECTIVIX_CONNECTIVITY_MATRIX_BASE_H