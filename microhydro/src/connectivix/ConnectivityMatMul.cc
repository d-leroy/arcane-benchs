#include "ConnectivityMatMul.h"
#include "CSR.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/Atomic.h"
#include "arcane/accelerator/GenericScanner.h"
#include "arcane/accelerator/LocalMemory.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLaunch.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/core/IndexedItemConnectivityView.h"
#include "arcane/mesh/ItemFamily.h"
#include "arcane/utils/MemoryUtils.h"
#include "define.h"

namespace ax = Arcane::Accelerator;

namespace Connectivix {

void ConnectivityMatMul::doMatMul() {
  const bool is_acc = ax::isAcceleratorPolicy(m_runner.executionPolicy());
  setup();
  m_meta->allocate(m_runner);
  m_meta->barrier();

  symbolicBinning();
  m_meta->barrier();

  symbolic();
  m_meta->barrier();

  numericBinning();
  m_meta->barrier();

  ax::Scanner<Int32>::exclusiveSum(&m_meta->get_run_queue(0), *m_C.rpt, *m_meta->scan_storage);
  m_meta->barrier();

  if (!is_acc) {
    m_C.rpt->swap(*m_meta->scan_storage);
  }

  m_C.nnz = (*m_meta->total_nnz)[0];
  if (is_acc) {
    m_C.col = new NumArray<Int32, MDDim1>(m_C.nnz, eMemoryRessource::Device);
  } else {
    m_C.col = new NumArray<Int32, MDDim1>(m_C.nnz, eMemoryRessource::Host);
  }

  numeric();
  m_meta->barrier();
}

// Computing number of products for each row,
// and max number of products in a single row.
void ConnectivityMatMul::setup() {
  const Int32 M = m_meta->M;

  auto command = makeCommand(m_meta->get_run_queue(0));

  auto row_flop_view = ax::viewOut(command, *m_C.rpt);
  auto max_flop_view = ax::viewInOut(command, *m_meta->max_row_nnz);
  auto arpt_view = ax::viewIn(command, *m_A.rpt);
  auto acol_view = ax::viewIn(command, *m_A.col);
  auto brpt_view = ax::viewIn(command, *m_B.rpt);

  ax::LocalMemory<Int32, 1> shared_max_row_flop(command, 1);

  ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, m_meta->M, 0, 0);

  if (!ax::isAcceleratorPolicy(m_runner.executionPolicy())) {
    row_flop_view[M] = 0;
  }

  command << RUNCOMMAND_LAUNCH(ctx, loop_range, shared_max_row_flop) {
    auto work_group = ctx.group();
    auto local_max_row_flop = shared_max_row_flop.span();

    if (work_group.isDevice()) {
      const bool is_rank0 = (work_group.activeWorkItemRankInGroup() == 0);
      if (is_rank0) {
        local_max_row_flop[0] = 0;
      }
      work_group.barrier();

      auto work_item = work_group.activeItem(0);
      Int32 i = work_item.linearIndex();
      Int32 row_flop = 0;
      Int32 j;
      Int32 acol;
      Int32 arow_start, arow_end;
      arow_start = arpt_view[i];
      arow_end = arpt_view[i + 1];
      for (j = arow_start; j < arow_end; ++j) {
        acol = acol_view[j];
        row_flop += brpt_view[acol + 1] - brpt_view[acol];
      }
      row_flop_view[i] = row_flop;
      ax::doAtomic<ax::eAtomicOperation::Max, Int32, Int32>(&local_max_row_flop[0], row_flop);

      work_group.barrier();

      if (is_rank0) {
        ax::doAtomic<ax::eAtomicOperation::Max, Int32, Int32>(max_flop_view[0], local_max_row_flop[0]);
      }
    } else {
      local_max_row_flop[0] = 0;

      for (Int32 item_rank = 0; item_rank < work_group.nbActiveItem(); ++item_rank) {
        const auto work_item = work_group.activeItem(item_rank);
        const Int32 i = work_item.linearIndex();
        const Int32 arow_start = arpt_view[i];
        const Int32 arow_end = arpt_view[i + 1];
        Int32 row_flop = 0;
        Int32 j;
        Int32 acol;
        for (j = arow_start; j < arow_end; ++j) {
          acol = acol_view[j];
          row_flop += brpt_view[acol + 1] - brpt_view[acol];
        }
        row_flop_view[i] = row_flop;
        ax::doAtomic<ax::eAtomicOperation::Max, Int32, Int32>(&local_max_row_flop[0], row_flop);
      }

      ax::doAtomic<ax::eAtomicOperation::Max, Int32, Int32>(max_flop_view[0], local_max_row_flop[0]);
    }
  };
}

} // namespace Connectivix