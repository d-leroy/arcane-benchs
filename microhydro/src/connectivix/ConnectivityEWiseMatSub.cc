#include "ConnectivityEWiseMatSub.h"
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/set_operations.h>

namespace Connectivix {

void ConnectivityEWiseMatSub::fillIndices(const CSR &m, Span<Int64> &out) {
  auto rptView = (*m.rpt).to1DSpan();
  auto colView = (*m.col).to1DSpan();

  thrust::for_each(thrust::counting_iterator<Int32>(0), thrust::counting_iterator<Int32>(m.nnz),
                   [rowOffset = rptView, colIndex = colView, outIndices = out, nrows = m.M, ncols = m.N] __device__(Int32 valueId) {
                     Int32 row = findNearestRowIdx(valueId, nrows, rowOffset);
                     Int32 col = colIndex[valueId];
                     Int64 index = Int64(row) * Int64(ncols) + Int64(col);
                     outIndices[valueId] = index;
                   });
}

void ConnectivityEWiseMatSub::doEWiseMatSub() {
  const ax::RunQueue queue = ax::makeQueue(m_runner);
  auto command = makeCommand(queue);

  auto aNvals = m_A.nnz;
  auto bNvals = m_B.nnz;
  auto worst = aNvals + bNvals;

  // Allocate memory for the worst case scenario
  NumArray<Int64, MDDim1> inputA(aNvals, eMemoryRessource::Device);
  NumArray<Int64, MDDim1> inputB(bNvals, eMemoryRessource::Device);
  NumArray<Int64, MDDim1> intersected(worst, eMemoryRessource::Device);

  auto inputASpan = inputA.to1DSpan();
  auto inputBSpan = inputB.to1DSpan();
  auto intersectedSpan = intersected.to1DSpan();

  fillIndices(m_A, inputASpan);
  fillIndices(m_B, inputBSpan);

  auto out = thrust::set_difference(thrust::device, inputASpan.begin(), inputASpan.end(), inputBSpan.begin(), inputBSpan.end(), intersectedSpan.begin());

  // Count result nvals count
  m_C.nnz = thrust::distance(intersectedSpan.begin(), out);

  NumArray<Int32, MDDim1> rowOffsetTmp(m_A.M + 1, eMemoryRessource::Device);

  auto rowOffsetTmpSpan = rowOffsetTmp.to1DSpan();
  thrust::fill(thrust::device, rowOffsetTmpSpan.begin(), rowOffsetTmpSpan.end(), 0);

  m_C.col = new NumArray<Int32, MDDim1>(m_C.nnz, eMemoryRessource::Device);
  auto colIndexSpan = (*m_C.col).to1DSpan();

  thrust::for_each(thrust::counting_iterator<Int32>(0), thrust::counting_iterator<Int32>(m_C.nnz),
                   [rowOffset = rowOffsetTmpSpan, colIndex = colIndexSpan, intersected = intersectedSpan, nrows = m_A.M, ncols = m_A.N] __device__(Int32 valueId) {
                     Int64 i = intersected[valueId];
                     Int32 row = Int32(i / ncols);
                     Int32 col = Int32(i % ncols);
                     ax::doAtomicAdd(rowOffset.ptrAt(row), 1);
                     colIndex[valueId] = col;
                   });

  m_C.rpt = new NumArray<Int32, MDDim1>(m_A.M + 1, eMemoryRessource::Device);
  thrust::exclusive_scan(thrust::device, rowOffsetTmpSpan.begin(), rowOffsetTmpSpan.end(), (*m_C.rpt).to1DSpan().begin(), 0, thrust::plus<Int32>());
}

} // namespace Connectivix