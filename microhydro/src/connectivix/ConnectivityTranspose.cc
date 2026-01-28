#include "ConnectivityTranspose.h"
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

namespace Connectivix {

void ConnectivityTranspose::doTranspose() {
  const ax::RunQueue queue = ax::makeQueue(m_runner);
  auto command = makeCommand(queue);

  auto nrows = m_A.M;
  auto ncols = m_A.N;
  auto nvals = m_A.nnz;

  m_AT.nnz = nvals;

  auto aRowIndices = m_A.rpt;
  auto aColIndices = m_A.col;

  // Create row index and column index to store res matrix
  // Note: transposed matrix has exact the same nnz
  NumArray<Int32, MDDim1> rowIndices(nvals, eMemoryRessource::Device);

  m_AT.col = new NumArray<Int32, MDDim1>(nvals, eMemoryRessource::Device);

  auto colIndicesSpan = m_AT.col->to1DSpan();

  // Copy col indices of a as row indices or aT
  rowIndices.copy(*aColIndices, queue);
  queue.barrier();
  auto rowIndicesSpan = rowIndices.to1DSpan();

  // Compute column indices of the aT for each value of a matrix
  thrust::for_each(thrust::counting_iterator<Int32>(0), thrust::counting_iterator<Int32>(nvals),
                   [aRowIndices = (*aRowIndices).to1DSpan(), nrows, colIndices = colIndicesSpan] __device__(Int32 valueId) {
                     Int32 rowId = findNearestRowIdx(valueId, nrows, aRowIndices);
                     colIndices[valueId] = rowId;
                   });

  // Sort row-col indices
  thrust::sort_by_key(thrust::device, rowIndicesSpan.begin(), rowIndicesSpan.end(), colIndicesSpan.begin());

  // Compute row offsets, based on row indices
  m_AT.rpt = new NumArray<Int32, MDDim1>(ncols + 1, eMemoryRessource::Device);
  m_AT.rpt->fill(0, queue);
  queue.barrier();

  NumArray<Int32, MDDim1> rowOffsetsTmp(ncols + 1, eMemoryRessource::Device);

  auto rowOffsetsTmpSpan = rowOffsetsTmp.to1DSpan();
  auto rowOffsetsSpan = m_AT.rpt->to1DSpan();

  thrust::for_each(thrust::device, rowIndicesSpan.begin(), rowIndicesSpan.end(), [rowOffsetsTmp = rowOffsetsTmpSpan] __device__(Int32 rowId) { ax::doAtomicAdd(rowOffsetsTmp.ptrAt(rowId), 1); });

  // Compute actual offsets
  thrust::exclusive_scan(thrust::device, rowOffsetsTmpSpan.begin(), rowOffsetsTmpSpan.end(), rowOffsetsSpan.begin(), 0, thrust::plus<Int32>());
}
} // namespace Connectivix
