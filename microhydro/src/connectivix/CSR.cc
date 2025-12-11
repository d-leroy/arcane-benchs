#include "CSR.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

namespace Connectivix {

void CSR::fromCoordinates(const Int32 *rows, const Int32 *cols, const Int32 nnz) {
  // TODO: we build the CSR on CPU, and copy the arrays on the GPU.
  this->nnz = nnz;
  // sort indices by (row,column)
  long *p = new long[nnz];
  Int32 *I_ = new Int32[nnz];
  col = new NumArray<Int32, MDDim1>(nnz);

  for (Int32 i = 0; i < nnz; ++i) {
    p[i] = (long int)N * rows[i] + cols[i];
  }

  std::sort(p, p + nnz);

  for (Int32 i = 0; i < nnz; ++i) {
    I_[i] = p[i] / N;
    (*col)[i] = p[i] % N;
  }
  delete[] p;

  // Converting from coordinates to compressed sparse row.
  rpt = new NumArray<Int32, MDDim1>(M + 1);
  rpt->fill(0);
  for (Int32 i = 0; i < nnz; ++i) {
    (*rpt)[I_[i] + 1]++; // => atomicAdd
  }
  for (Int32 i = 1; i <= M; ++i) {
    (*rpt)[i] += (*rpt)[i - 1];
  }

  delete[] I_;
}

CSR *CSR::transpose() const {
  CSR *result = new CSR(N, M);
  result->nnz = nnz;
  result->rpt = new NumArray<Int32, MDDim1>(N + 1);
  result->col = new NumArray<Int32, MDDim1>(nnz);

  long *p = new long[nnz];
  Int32 *I_ = new Int32[nnz];

  // Converting from compressed sparse row to transposed coordinates.
  for (Int32 i = 0; i < M; ++i) {
    for (Int32 j = (*rpt)[i]; j < (*rpt)[i + 1]; ++j) {
      p[j] = (long int)M * (*col)[j] + i;
    }
  }

  std::sort(p, p + nnz);
  for (Int32 i = 0; i < nnz; ++i) {
    I_[i] = p[i] / M;
    (*result->col)[i] = p[i] % M;
  }
  delete[] p;

  // Converting from coordinates back to compressed sparse row.
  result->rpt->fill(0);
  for (Int32 i = 0; i < nnz; ++i) {
    (*result->rpt)[I_[i] + 1]++;
  }
  for (Int32 i = 1; i <= N; ++i) {
    (*result->rpt)[i] += (*result->rpt)[i - 1];
  }

  delete[] I_;

  return result;
}

std::string CSR::printRows() const {
  std::stringstream sstream;
  sstream << "Rows (" << M << "):" << std::endl;
  for (Int32 i = 0; i < M + 1; ++i) {
    sstream << (*rpt)[i] << " ";
  }
  sstream << std::endl;
  return sstream.str();
}

std::string CSR::printCols() const {
  std::stringstream sstream;
  sstream << "Cols (" << nnz << "):" << std::endl;
  for (Int32 i = 0; i < nnz; ++i) {
    sstream << (*col)[i] << " ";
  }
  sstream << std::endl;
  return sstream.str();
}

std::string CSR::printMatrix() const {
  std::stringstream sstream;
  sstream << "Matrix:" << std::endl;
  for (Int32 row = 0; row < M; row++) {
    Int32 begin = (*rpt)[row];
    Int32 end = (*rpt)[row + 1];

    for (Int32 i = begin; i < end; i++) {
      sstream << row << " " << (*col)[i] << std::endl;
    }
  }
  return sstream.str();
}

void CSR::dumpMatrix(std::ofstream &file) const {
  file << "%%MatrixMarket matrix coordinate pattern general" << std::endl;
  file << M << " " << N << " " << (*rpt)[M] << std::endl;

  for (Int32 row = 0; row < M; row++) {
    Int32 begin = (*rpt)[row];
    Int32 end = (*rpt)[row + 1];

    for (Int32 i = begin; i < end; i++) {
      file << row << " " << (*col)[i] << std::endl;
    }
  }
}

CSR::~CSR() {
  delete rpt;
  delete col;
}

} // namespace Connectivix