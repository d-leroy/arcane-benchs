#ifndef CONNECTIVIX_ITEM_ITERATOR_H
#define CONNECTIVIX_ITEM_ITERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Connectivix {

template <typename ItemType> struct ItemIterator {
public:
  using ItemTypeType = ItemType;
  using ItemLocalId = typename ItemType::LocalIdType;

  ItemLocalId *begin_ptr;
  ItemLocalId *end_ptr;

  ARCCORE_HOST_DEVICE ItemLocalId *begin() { return begin_ptr; }
  ARCCORE_HOST_DEVICE ItemLocalId *end() { return end_ptr; }
};
} // namespace Connectivix

#endif // CONNECTIVIX_CONNECTIVITY_MATRIX_BASE_H