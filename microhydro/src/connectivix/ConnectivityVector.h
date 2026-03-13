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
#include <concepts>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;
using namespace Arcane;
namespace ax = Arcane::Accelerator;

namespace Connectivix {

template <typename V>
concept ConnectivityVectorC = /*std::ranges::range<const V> &&*/ requires(const V &v, typename V::ItemType x, Int32 i) {
  typename V::ItemType;
  { v.size() } -> std::convertible_to<Int32>;
  { v[i] } -> std::convertible_to<const typename V::ItemType>;
  { v.sorted_at(i) } -> std::convertible_to<const typename V::ItemType>;
  { v.itemIndex(x) } -> std::convertible_to<int>;
};

template <ConnectivityVectorC L, ConnectivityVectorC R>
  requires std::same_as<typename L::ItemType, typename R::ItemType>
struct LazyVectorIntersection {
  using ItemType = typename L::ItemType;

  const L *m_lhs;
  const R *m_rhs;

  ARCCORE_HOST_DEVICE LazyVectorIntersection(const L &lhs, const R &rhs) : m_lhs(&lhs), m_rhs(&rhs) {}

  struct iterator {
    const LazyVectorIntersection *parent;
    Int32 i_lhs;
    Int32 i_rhs;

    ARCCORE_HOST_DEVICE iterator(const LazyVectorIntersection *p, Int32 a, Int32 b) : parent(p), i_lhs(a), i_rhs(b) {
      advance();
    }

    ARCCORE_HOST_DEVICE void advance() {
      auto &lhs = *parent->m_lhs;
      auto &rhs = *parent->m_rhs;

      while (i_lhs < lhs.size() && i_rhs < rhs.size()) {
        const auto &va = lhs.sorted_at(i_lhs);
        const auto &vb = rhs.sorted_at(i_rhs);

        if (va < vb)
          ++i_lhs;
        else if (vb < va)
          ++i_rhs;
        else
          return;
      }

      i_lhs = lhs.size();
    }

    ARCCORE_HOST_DEVICE const ItemType operator*() const {
      return parent->m_lhs->sorted_at(i_lhs);
    }

    ARCCORE_HOST_DEVICE iterator &operator++() {
      ++i_lhs;
      ++i_rhs;
      advance();
      return *this;
    }

    ARCCORE_HOST_DEVICE bool operator==(const iterator &o) const {
      return i_lhs == o.i_lhs;
    }

    ARCCORE_HOST_DEVICE bool operator!=(const iterator &o) const {
      return !(*this == o);
    }
  };

  ARCCORE_HOST_DEVICE auto begin() const {
    return iterator(this, 0, 0);
  }

  ARCCORE_HOST_DEVICE auto end() const {
    return iterator(this, m_lhs->size(), 0);
  }
};

template <ConnectivityVectorC L, ConnectivityVectorC R>
  requires std::same_as<typename L::ItemType, typename R::ItemType>
struct LazyVectorSubtraction {
  using ItemType = typename L::ItemType;

  const L *m_lhs;
  const R *m_rhs;

  ARCCORE_HOST_DEVICE LazyVectorSubtraction(const L &lhs, const R &rhs) : m_lhs(&lhs), m_rhs(&rhs) {}

  struct iterator {
    const LazyVectorSubtraction *parent;
    Int32 i_lhs;
    Int32 i_rhs;

    ARCCORE_HOST_DEVICE iterator(const LazyVectorSubtraction *p, Int32 a, Int32 b) : parent(p), i_lhs(a), i_rhs(b) {
      advance();
    }

    ARCCORE_HOST_DEVICE void advance() {
      auto &lhs = *parent->m_lhs;
      auto &rhs = *parent->m_rhs;

      while (i_lhs < lhs.size()) {
        const auto &va = lhs.sorted_at(i_lhs);
        const auto &vb = rhs.sorted_at(i_rhs);

        if (i_rhs == rhs.size() || va < vb) {
          return;
        } else if (va == vb) {
          ++i_lhs;
          ++i_rhs;
        } else {
          ++i_rhs;
        }
      }

      i_lhs = lhs.size();
    }

    ARCCORE_HOST_DEVICE const ItemType &operator*() const {
      return parent->m_lhs->sorted_at(i_lhs);
    }

    ARCCORE_HOST_DEVICE iterator &operator++() {
      ++i_lhs;
      ++i_rhs;
      advance();
      return *this;
    }

    ARCCORE_HOST_DEVICE bool operator==(const iterator &o) const {
      return i_lhs == o.i_lhs;
    }

    ARCCORE_HOST_DEVICE bool operator!=(const iterator &o) const {
      return !(*this == o);
    }
  };

  ARCCORE_HOST_DEVICE auto begin() const {
    return iterator(this, 0, 0);
  }

  ARCCORE_HOST_DEVICE auto end() const {
    return iterator(this, m_lhs->size(), 0);
  }
};

template <typename T> struct ConnectivityVector {
  using ItemType = T;

  Span<const Int32> m_items;

  ARCCORE_HOST_DEVICE ConnectivityVector(Span<const Int32> items) : m_items(items) {}

  ARCCORE_HOST_DEVICE Int32 size() const {
    return m_items.size();
  }

  ARCCORE_HOST_DEVICE const ItemType operator[](Int32 i) const {
    return ItemType(m_items[i]);
  }

  ARCCORE_HOST_DEVICE const ItemType sorted_at(Int32 i) const {
    return ItemType(m_items[i]);
  }

  ARCCORE_HOST_DEVICE int itemIndex(const ItemType &item) const {
    Int32 result = 0;
    for (Int32 k = 0; k < m_items.size(); ++k) {
      result += (unsigned int)(m_items[k] - item) >> 31;
    }
    Int32 found = (unsigned int)(result - item) >> 31;
    return result * found + (1 - found) * -1;
  }

  struct iterator {
    const ConnectivityVector *parent = nullptr;
    Int32 idx = 0;

    ARCCORE_HOST_DEVICE iterator() = default;
    ARCCORE_HOST_DEVICE iterator(const ConnectivityVector *p, Int32 start) : parent(p), idx(start) {}

    ARCCORE_HOST_DEVICE ItemType operator*() const {
      return ItemType(parent->m_items[idx]);
    }

    ARCCORE_HOST_DEVICE iterator &operator++() {
      ++idx;
      return *this;
    }

    ARCCORE_HOST_DEVICE iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    ARCCORE_HOST_DEVICE bool operator==(const iterator &o) const {
      return parent == o.parent && idx == o.idx;
    }

    ARCCORE_HOST_DEVICE bool operator!=(const iterator &o) const {
      return !(*this == o);
    }
  };

  ARCCORE_HOST_DEVICE auto begin() const {
    return iterator(this, 0);
  }
  ARCCORE_HOST_DEVICE auto end() const {
    return iterator(this, m_items.size());
  }

  template <ConnectivityVectorC Other>
    requires std::same_as<ItemType, typename Other::ItemType>
  ARCCORE_HOST_DEVICE auto intersect(const Other &other) const {
    return LazyVectorIntersection<ConnectivityVector<ItemType>, Other>(*this, other);
  }

  template <ConnectivityVectorC Other>
    requires std::same_as<ItemType, typename Other::ItemType>
  ARCCORE_HOST_DEVICE auto subtract(const Other &other) const {
    return LazyVectorSubtraction(*this, other);
  }
};

template <typename T> struct OrderedConnectivityVector {
  using ItemType = T;

  Span<const Int32> m_items;
  Span<const Int32> m_order;

  ARCCORE_HOST_DEVICE OrderedConnectivityVector(Span<const Int32> items, Span<const Int32> order) : m_items(items), m_order(order) {}

  ARCCORE_HOST_DEVICE Int32 size() const {
    return m_order.size();
  }

  ARCCORE_HOST_DEVICE const ItemType operator[](Int32 i) const {
    return ItemType(m_items[m_order[i]]);
  }

  ARCCORE_HOST_DEVICE const ItemType sorted_at(Int32 i) const {
    return ItemType(m_items[i]);
  }

  ARCCORE_HOST_DEVICE int itemIndex(const ItemType &item) const {
    Int32 result = 0;
    for (Int32 k = 0; k < m_items.size(); ++k) {
      result += (unsigned int)(m_items[k] - item) >> 31;
    }
    Int32 found = (unsigned int)(result - item) >> 31;
    return m_order[result * found + (1 - found) * -1];
  }

  struct iterator {
    const OrderedConnectivityVector *parent = nullptr;
    Int32 idx = 0;

    ARCCORE_HOST_DEVICE iterator() = default;
    ARCCORE_HOST_DEVICE iterator(const OrderedConnectivityVector *p, Int32 start) : parent(p), idx(start) {}

    ARCCORE_HOST_DEVICE ItemType operator*() const {
      return ItemType(parent->m_items[parent->m_order[idx]]);
    }

    ARCCORE_HOST_DEVICE iterator &operator++() {
      ++idx;
      return *this;
    }

    ARCCORE_HOST_DEVICE iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    ARCCORE_HOST_DEVICE bool operator==(const iterator &o) const {
      return parent == o.parent && idx == o.idx;
    }

    ARCCORE_HOST_DEVICE bool operator!=(const iterator &o) const {
      return !(*this == o);
    }
  };

  ARCCORE_HOST_DEVICE auto begin() const {
    return iterator(this, 0);
  }
  ARCCORE_HOST_DEVICE auto end() const {
    return iterator(this, m_order.size());
  }

  template <ConnectivityVectorC Other>
    requires std::same_as<ItemType, typename Other::ItemType>
  ARCCORE_HOST_DEVICE auto intersect(const Other &other) const {
    return LazyVectorIntersection(*this, other);
  }

  template <ConnectivityVectorC Other>
    requires std::same_as<ItemType, typename Other::ItemType>
  ARCCORE_HOST_DEVICE auto subtract(const Other &other) const {
    return LazyVectorSubtraction(*this, other);
  }
};

} // namespace Connectivix

#endif // CONNECTIVIX_CONNECTIVITY_VECTOR_H