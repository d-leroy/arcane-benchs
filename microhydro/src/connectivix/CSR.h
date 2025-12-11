#ifndef CONNECTIVIX_CSR_H
#define CONNECTIVIX_CSR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"
#include <string>
#include <vector>

#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/utils/MemoryUtils.h"

using namespace Arccore;
using namespace Arcane;

namespace Connectivix {

struct CSR {

  CSR(const Int32 M, const Int32 N) : M(M), N(N), nnz(0), rpt(nullptr), col(nullptr) {}
  ~CSR();

  void allocate();

  void fromCoordinates(const Int32 *rows, const Int32 *cols, const Int32 nnz);

  CSR *transpose() const;

  std::string printRows() const;
  std::string printCols() const;
  std::string printMatrix() const;

  void dumpMatrix(std::ofstream &file) const;

  Int32 M;
  Int32 N;
  Int32 nnz;

  NumArray<Int32, MDDim1> *rpt;
  NumArray<Int32, MDDim1> *col;
};

} // namespace Connectivix

#endif // CONNECTIVIX_CSR_H