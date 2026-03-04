#ifndef CONNECTIVIX_CONNECTIVITY_VECTOR_H
#define CONNECTIVIX_CONNECTIVITY_VECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "CSR.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/Atomic.h"
#include "arcane/accelerator/LocalMemory.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLaunch.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "connectivix/ConnectivityEWiseMatMul.h"
#include "connectivix/ConnectivityEWiseMatSub.h"
#include "define.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;
using namespace Arcane;
namespace ax = Arcane::Accelerator;

namespace Connectivix {

template <typename ItemLocalId> class ConnectivityVectorIntersectionView;
template <typename ItemLocalId> class ConnectivityVectorSubtractionView;

template <typename ItemLocalId> class ConnectivityVectorView {

public:
  class ConnectivityVectorIterator {
    // using iterator_type = ConnectivityVectorView<ItemLocalId>;
    using iterator_type = Arcane::ArrayIterator<const int *>;

    iterator_type it, it_begin, it_end;

  public:
    ARCCORE_HOST_DEVICE ConnectivityVectorIterator(iterator_type begin, iterator_type end) : it(begin), it_begin(begin), it_end(end) {}

    ARCCORE_HOST_DEVICE ConnectivityVectorIterator &operator++() {
      ++it;
      return *this;
    }

    ARCCORE_HOST_DEVICE ItemLocalId operator*() const {
      return ItemLocalId(*it);
    }

    ARCCORE_HOST_DEVICE bool operator!=(const ConnectivityVectorIterator &other) const {
      return it != other.it;
    }

    ARCCORE_HOST_DEVICE bool operator==(const ConnectivityVectorIterator &other) const {
      return it == other.it;
    }
  };

public:
  ARCCORE_HOST_DEVICE ConnectivityVectorView(const ax::NumArrayInView<Int32, MDDim1> &items, const Int32 nb_vals) : nb_vals(nb_vals) {
    this->items = Span<const Int32>(items.to1DSpan());
  }
  ARCCORE_HOST_DEVICE ConnectivityVectorView(Span<const Int32> items, const Int32 nb_vals) : items(items), nb_vals(nb_vals) {}
  ARCCORE_HOST_DEVICE ConnectivityVectorView(Int32 *ptr, const Int32 size, const Int32 nb_vals) : nb_vals(nb_vals) {
    this->items = Span<const Int32>(ptr, size);
  }

protected:
  Span<const Int32> items;
  const Int32 nb_vals;

public:
  ARCCORE_HOST_DEVICE inline Int32 itemIndex(ItemLocalId item) const {
    Int32 result = 0;
    for (Int32 k = 0; k < items.size(); ++k) {
      result += (unsigned int)(items[k] - item) >> 31;
    }
    Int32 found = (unsigned int)(result - item) >> 31;
    return result * found + (1 - found) * -1;
  }

  ARCCORE_HOST_DEVICE inline ConnectivityVectorIntersectionView<ItemLocalId> intersect(ConnectivityVectorView<ItemLocalId> &other) const {
    return ConnectivityVectorIntersectionView<ItemLocalId>(*this, other);
  }

  ARCCORE_HOST_DEVICE inline ConnectivityVectorSubtractionView<ItemLocalId> subtract(ConnectivityVectorView<ItemLocalId> &other) const {
    return ConnectivityVectorSubtractionView<ItemLocalId>(*this, other);
  }

  ARCCORE_HOST_DEVICE inline Int32 size() const {
    return nb_vals;
  }

  constexpr ARCCORE_HOST_DEVICE auto begin() const noexcept {
    return ConnectivityVectorIterator(items.begin(), items.end());
  }

  constexpr ARCCORE_HOST_DEVICE auto end() const noexcept {
    return ConnectivityVectorIterator(items.end(), items.end());
  }

  constexpr ARCCORE_HOST_DEVICE ItemLocalId operator[](Int32 i) const {
    return ItemLocalId(items[i]);
  }
};

template <typename ItemLocalId> class ConnectivityVectorIntersectionView {

public:
  class ConnectivityVectorIntersectionIterator {
    using iterator_type = typename ConnectivityVectorView<ItemLocalId>::ConnectivityVectorIterator;
    // using iterator_type = Arcane::ArrayIterator<const int *>;

    iterator_type it_a, it_a_begin, it_a_end;
    iterator_type it_b, it_b_begin, it_b_end;
    ItemLocalId current_value;

  public:
    ARCCORE_HOST_DEVICE ConnectivityVectorIntersectionIterator(iterator_type a_begin, iterator_type a_end, iterator_type b_begin, iterator_type b_end)
        : it_a(a_begin), it_a_begin(a_begin), it_a_end(a_end), it_b(b_begin), it_b_begin(b_begin), it_b_end(b_end) {
      advance_to_next();
    }

    ARCCORE_HOST_DEVICE void advance_to_next() {
      while (it_a != it_a_end && it_b != it_b_end) {
        if (*it_a == *it_b) {
          current_value = *it_a;
          return;
        } else if (*it_a < *it_b) {
          ++it_a;
        } else {
          ++it_b;
        }
      }
      // If no more common elements
      current_value = ItemLocalId(-1);
    }

    ARCCORE_HOST_DEVICE Int32 compute_size() {
      Int32 result = 0;
      while (it_a != it_a_end && it_b != it_b_end) {
        if (*it_a == *it_b) {
          ++result;
          ++it_a;
          ++it_b;
        } else if (*it_a < *it_b) {
          ++it_a;
        } else {
          ++it_b;
        }
      }
      return result;
    }

    ARCCORE_HOST_DEVICE ConnectivityVectorIntersectionIterator &operator++() {
      ++it_a;
      ++it_b;
      advance_to_next();
      return *this;
    }

    ARCCORE_HOST_DEVICE ItemLocalId operator*() const {
      return current_value;
    }

    ARCCORE_HOST_DEVICE bool operator!=(const ConnectivityVectorIntersectionIterator &other) const {
      return (it_a != other.it_a) && (it_b != other.it_b);
    }
  };

private:
  const ConnectivityVectorView<ItemLocalId> &items_a;
  const ConnectivityVectorView<ItemLocalId> &items_b;

public:
  ARCCORE_HOST_DEVICE ConnectivityVectorIntersectionView(const ConnectivityVectorView<ItemLocalId> &items_a, const ConnectivityVectorView<ItemLocalId> &items_b) : items_a(items_a), items_b(items_b) {}

  ARCCORE_HOST_DEVICE Arcane::Int32 size() const {
    return ConnectivityVectorIntersectionIterator(items_a.begin(), items_a.end(), items_b.begin(), items_b.end()).compute_size();
  }

  constexpr ARCCORE_HOST_DEVICE ConnectivityVectorIntersectionIterator begin() const noexcept {
    return ConnectivityVectorIntersectionIterator(items_a.begin(), items_a.end(), items_b.begin(), items_b.end());
  }

  constexpr ARCCORE_HOST_DEVICE ConnectivityVectorIntersectionIterator end() const noexcept {
    return ConnectivityVectorIntersectionIterator(items_a.end(), items_a.end(), items_b.end(), items_b.end());
  }
};

template <typename ItemLocalId> class ConnectivityVectorSubtractionView {

public:
  class ConnectivityVectorSubtractionIterator {
    // using iterator_type = ConnectivityVectorView<ItemLocalId>;
    using iterator_type = typename ConnectivityVectorView<ItemLocalId>::ConnectivityVectorIterator;

    iterator_type it_a, it_a_begin, it_a_end;
    iterator_type it_b, it_b_begin, it_b_end;
    ItemLocalId current_value;

  public:
    ARCCORE_HOST_DEVICE ConnectivityVectorSubtractionIterator(iterator_type a_begin, iterator_type a_end, iterator_type b_begin, iterator_type b_end)
        : it_a(a_begin), it_a_begin(a_begin), it_a_end(a_end), it_b(b_begin), it_b_begin(b_begin), it_b_end(b_end) {
      advance_to_next();
    }

    ARCCORE_HOST_DEVICE void advance_to_next() {
      while (it_a != it_a_end) {
        if (it_b == it_b_end || *it_a < *it_b) {
          current_value = *it_a;
          return;
        } else if (*it_a == *it_b) {
          ++it_a;
          ++it_b;
        } else {
          ++it_b;
        }
      }
      // If no more common elements
      current_value = ItemLocalId(-1);
    }

    ARCCORE_HOST_DEVICE Int32 compute_size() {
      Int32 result = 0;
      while (it_a != it_a_end) {
        if (it_b == it_b_end || *it_a < *it_b) {
          ++result;
        } else if (*it_a == *it_b) {
          ++it_a;
          ++it_b;
        } else {
          ++it_b;
        }
      }
      return result;
    }

    ARCCORE_HOST_DEVICE ConnectivityVectorSubtractionIterator &operator++() {
      ++it_a;
      advance_to_next();
      return *this;
    }

    ARCCORE_HOST_DEVICE inline ItemLocalId operator*() const {
      return current_value;
    }

    ARCCORE_HOST_DEVICE inline bool operator!=(const ConnectivityVectorSubtractionIterator &other) const {
      return it_a != other.it_a;
    }
  };

private:
  const ConnectivityVectorView<ItemLocalId> &items_a;
  const ConnectivityVectorView<ItemLocalId> &items_b;

public:
  ARCCORE_HOST_DEVICE ConnectivityVectorSubtractionView(const ConnectivityVectorView<ItemLocalId> &items_a, const ConnectivityVectorView<ItemLocalId> &items_b) : items_a(items_a), items_b(items_b) {}

  ARCCORE_HOST_DEVICE Arcane::Int32 size() const {
    return ConnectivityVectorSubtractionIterator(items_a.begin(), items_a.end(), items_b.begin(), items_b.end()).compute_size();
  }

  constexpr ARCCORE_HOST_DEVICE ConnectivityVectorSubtractionIterator begin() const noexcept {
    return ConnectivityVectorSubtractionIterator(items_a.begin(), items_a.end(), items_b.begin(), items_b.end());
  }

  constexpr ARCCORE_HOST_DEVICE ConnectivityVectorSubtractionIterator end() const noexcept {
    return ConnectivityVectorSubtractionIterator(items_a.end(), items_a.end(), items_b.end(), items_b.end());
  }
};

} // namespace Connectivix

#endif // CONNECTIVIX_CONNECTIVITY_VECTOR_H