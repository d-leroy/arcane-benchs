#ifndef CONNECTIVIX_CONNECTIVITY_EWISE_MAT_MUL_H
#define CONNECTIVIX_CONNECTIVITY_EWISE_MAT_MUL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "CSR.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/Atomic.h"
#include "arcane/accelerator/LocalMemory.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLaunch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;
using namespace Arcane;

namespace ax = Arcane::Accelerator;

namespace Connectivix {

class ConnectivityEWiseMatMul {

  static __device__ Int32 findNearestRowIdx(Int32 value, Int32 range, const Span<const Int32> rowIndex) {
    Int32 left = 0;
    Int32 right = range - 1;

    while (left <= right) {
      Int32 i = (left + right) / 2;

      if (value < rowIndex[i]) {
        // value is not in =>row i range
        right = i - 1;
      } else if (value < rowIndex[i + 1]) {
        // value is in =>row i range and value actually in row i
        return i;
      } else {
        // value is in =>row i+1 range
        left = i + 1;
      }
    }

    // never goes here since kron row index always has value index
    return range;
  }

public:
  ConnectivityEWiseMatMul(const CSR &A, const CSR &B, CSR &C, ax::Runner &runner) : m_A(A), m_B(B), m_C(C), m_runner(runner) {};

public:
  void doEWiseMatMul();

public:
  void fillIndices(const CSR &m, Span<Int64> &out);

private:
  ax::Runner &m_runner;
  const CSR &m_A;
  const CSR &m_B;
  CSR &m_C;
};
} // namespace Connectivix
#endif // CONNECTIVIX_CONNECTIVITY_EWISE_MAT_MUL_H