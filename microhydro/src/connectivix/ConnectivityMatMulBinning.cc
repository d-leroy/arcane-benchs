#include "CSR.h"
#include "ConnectivityMatMul.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/Atomic.h"
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

void ConnectivityMatMul::symbolicBinning() {
  // We use total_nnz to store max row products.
  if ((*m_meta->total_nnz)[0] <= 32) {
    symbolicBinningSmall();

    (*m_meta->bin_size)[0] = m_meta->M;
    for (int i = 1; i < NUM_BIN; ++i) {
      (*m_meta->bin_size)[i] = 0;
    }
    (*m_meta->bin_offset)[0] = 0;
    for (int i = 1; i < NUM_BIN; i++) {
      (*m_meta->bin_offset)[i] = m_meta->M;
    }
  } else {
    symbolicBinningFirstPass();

    (*m_meta->bin_offset)[0] = 0;
    for (int i = 0; i < NUM_BIN - 1; ++i) {
      (*m_meta->bin_offset)[i + 1] = (*m_meta->bin_offset)[i] + (*m_meta->bin_size)[i];
    }

    symbolicBinningSecondPass();
  }
}

void ConnectivityMatMul::symbolicBinningSmall() {
  Int32 GS = div_up(m_meta->M, 1024);
  auto queue = m_meta->run_queues[0].get();
  auto command = makeCommand(queue);
  auto bins_view = ax::viewOut(command, *m_meta->bins);

  command << RUNCOMMAND_LOOP1(iter, m_meta->M) {
    auto [i] = iter();
    bins_view[i] = i;
  };

  queue->barrier();
}

void ConnectivityMatMul::symbolicBinningFirstPass() {
  const Int32 M = m_meta->M;

  auto queue = m_meta->run_queues[0].get();
  auto command = makeCommand(m_meta->run_queues[0].get());

  auto row_flop_view = ax::viewIn(command, *m_C.rpt);
  auto bin_offset_view = ax::viewInOut(command, *m_meta->bin_offset);
  auto bin_size_view = ax::viewInOut(command, *m_meta->bin_size);
  auto bins_view = ax::viewOut(command, *m_meta->bins);

  ax::LocalMemory<Int32, NUM_BIN> shared_bin_size(command, NUM_BIN);

  ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, M, 0, 0);

  command << RUNCOMMAND_LAUNCH(ctx, loop_range, shared_bin_size) {
    auto work_group = ctx.group();
    auto local_bin_size = shared_bin_size.span();
    constexpr Int32 range[NUM_BIN] = {32, 512, 1024, 2048, 4096, 8192, 12287, INT_MAX};
    Int32 row_nnz, j;

    if (work_group.isDevice()) {
      const Int32 item_rank = work_group.activeWorkItemRankInGroup();
      auto work_item = work_group.activeItem(0);
      Int32 i = work_item.linearIndex();

      if (item_rank < NUM_BIN) {
        local_bin_size[item_rank] = 0;
      }

      work_group.barrier();
      if (i < M) {
        row_nnz = row_flop_view[i];
        // #pragma unroll
        for (j = 0; j < NUM_BIN; ++j) {
          if (row_nnz <= range[j]) {
            ax::doAtomicAdd(&local_bin_size[j], 1);
            break;
          }
        }
      }

      work_group.barrier();

      if (item_rank < NUM_BIN) {
        ax::doAtomicAdd(bin_size_view[item_rank], local_bin_size[item_rank]);
      }
    } else {
      for (Int32 b = 0; b < NUM_BIN; ++b) {
        local_bin_size[b] = 0;
      }

      for (Int32 g = 0; g < work_group.nbActiveItem(); ++g) {
        auto work_item = work_group.activeItem(g);
        Int32 i = work_item.linearIndex() + g;

        if (i < M) {
          row_nnz = row_flop_view[i];
          // #pragma unroll
          for (j = 0; j < NUM_BIN; ++j) {
            if (row_nnz <= range[j]) {
              ax::doAtomicAdd(&local_bin_size[j], 1);
              break;
            }
          }
        }
      }

      for (Int32 b = 0; b < NUM_BIN; ++b) {
        ax::doAtomicAdd(bin_size_view[b], local_bin_size[b]);
      }
    }
  };

  queue->barrier();
}

void ConnectivityMatMul::symbolicBinningSecondPass() {
  const Int32 M = m_meta->M;

  auto queue = m_meta->run_queues[0].get();
  auto command = makeCommand(m_meta->run_queues[0].get());

  auto row_flop_view = ax::viewIn(command, *m_C.rpt);
  auto bin_offset_view = ax::viewInOut(command, *m_meta->bin_offset);
  auto bin_size_view = ax::viewInOut(command, *m_meta->bin_size);
  auto bins_view = ax::viewOut(command, *m_meta->bins);

  ax::LocalMemory<Int32, NUM_BIN> shared_bin_size(command, NUM_BIN);
  ax::LocalMemory<Int32, NUM_BIN> shared_bin_offset(command, NUM_BIN);

  ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, M, 0, 0);

  command << RUNCOMMAND_LAUNCH(ctx, loop_range, shared_bin_size, shared_bin_offset) {
    auto work_group = ctx.group();
    auto local_bin_size = shared_bin_size.span();
    auto local_bin_offset = shared_bin_offset.span();
    constexpr Int32 range[NUM_BIN] = {32, 512, 1024, 2048, 4096, 8192, 12287, INT_MAX};
    Int32 row_nnz, j;

    if (work_group.isDevice()) {
      const Int32 item_rank = work_group.activeWorkItemRankInGroup();
      auto work_item = work_group.activeItem(0);
      Int32 i = work_item.linearIndex();

      if (item_rank < NUM_BIN) {
        local_bin_size[item_rank] = 0;
      }

      work_group.barrier();

      if (i < M) {
        row_nnz = row_flop_view[i];
        // #pragma unroll
        for (j = 0; j < NUM_BIN; ++j) {
          if (row_nnz <= range[j]) {
            ax::doAtomicAdd(&local_bin_size[j], 1);
            break;
          }
        }
      }

      work_group.barrier();

      if (item_rank < NUM_BIN) {
        local_bin_offset[item_rank] = ax::doAtomicAdd(bin_size_view[item_rank], local_bin_size[item_rank]);
        local_bin_offset[item_rank] += bin_offset_view[item_rank];
        local_bin_size[item_rank] = 0;
      }
      work_group.barrier();

      Int32 index;
      if (i < M) {
        // #pragma unroll
        for (j = 0; j < NUM_BIN; ++j) {
          if (row_nnz <= range[j]) {
            index = ax::doAtomicAdd(&local_bin_size[j], 1);
            bins_view[local_bin_offset[j] + index] = i;
            return;
          }
        }
      }
    } else {
      for (Int32 b = 0; b < NUM_BIN; ++b) {
        local_bin_size[b] = 0;
      }

      for (Int32 g = 0; g < work_group.nbActiveItem(); ++g) {
        auto work_item = work_group.activeItem(g);
        Int32 i = work_item.linearIndex() + g;

        if (i < M) {
          row_nnz = row_flop_view[i];
          // #pragma unroll
          for (j = 0; j < NUM_BIN; ++j) {
            if (row_nnz <= range[j]) {
              ax::doAtomicAdd(&local_bin_size[j], 1);
              break;
            }
          }
        }
      }

      for (Int32 b = 0; b < NUM_BIN; ++b) {
        local_bin_offset[b] = ax::doAtomicAdd(bin_size_view[b], local_bin_size[b]);
        local_bin_offset[b] += bin_offset_view[b];
        local_bin_size[b] = 0;
      }

      for (Int32 g = 0; g < work_group.nbActiveItem(); ++g) {
        auto work_item = work_group.activeItem(g);
        Int32 i = work_item.linearIndex() + g;

        Int32 index;
        if (i < M) {
          row_nnz = row_flop_view[i];
          // #pragma unroll
          for (j = 0; j < NUM_BIN; ++j) {
            if (row_nnz <= range[j]) {
              index = ax::doAtomicAdd(&local_bin_size[j], 1);
              bins_view[local_bin_offset[j] + index] = i;
              return;
            }
          }
        }
      }
    }
  };

  queue->barrier();
}

void ConnectivityMatMul::numericBinning() {
  numericBinningFirstPass();

  if ((*m_meta->max_row_nnz)[0] <= 32) {
    numericBinningSmall();

    (*m_meta->bin_size)[0] = m_meta->M;
    for (int i = 1; i < NUM_BIN; ++i) {
      (*m_meta->bin_size)[i] = 0;
    }
    (*m_meta->bin_offset)[0] = 0;
    for (int i = 1; i < NUM_BIN; i++) {
      (*m_meta->bin_offset)[i] = m_meta->M;
    }
  } else {
    (*m_meta->bin_offset)[0] = 0;
    for (int i = 0; i < NUM_BIN - 1; ++i) {
      (*m_meta->bin_offset)[i + 1] = (*m_meta->bin_offset)[i] + (*m_meta->bin_size)[i];
    }

    numericBinningSecondPass();
  }
}

void ConnectivityMatMul::numericBinningFirstPass() {
  const Int32 M = m_meta->M;
  const Int32 group_size = 512; // FIXME! 1024;
  const Int32 nb_groups = div_up(M, group_size);

  auto queue = m_meta->run_queues[0].get();
  auto command = makeCommand(m_meta->run_queues[0].get());

  auto row_nnz_view = ax::viewInOut(command, *m_C.rpt);
  auto bin_size_view = ax::viewInOut(command, *m_meta->bin_size);
  auto total_nnz_view = ax::viewInOut(command, *m_meta->total_nnz);
  auto max_row_nnz_view = ax::viewInOut(command, *m_meta->max_row_nnz);

  ax::LocalMemory<Int32, NUM_BIN> shared_bin_size(command, NUM_BIN);
  ax::LocalMemory<Int32, 1> shared_local_nnz(command, 1);
  ax::LocalMemory<Int32, 1> shared_max_row_nnz(command, 1);
  max_row_nnz_view[0] = 0;

  ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, nb_groups * group_size, nb_groups, group_size);

  command << RUNCOMMAND_LAUNCH(ctx, loop_range, shared_bin_size, shared_local_nnz, shared_max_row_nnz) {
    auto work_group = ctx.group();
    auto local_bin_size = shared_bin_size.span();
    auto local_local_nnz = shared_local_nnz.span();
    auto local_max_row_nnz = shared_max_row_nnz.span();

    constexpr Int32 range[NUM_BIN] = {31, 255, 511, 1022, 2047, 4095, 8191, INT_MAX};
    Int32 row_nnz, j;

    if (work_group.isDevice()) {
      const Int32 item_rank = work_group.activeWorkItemRankInGroup();
      if (item_rank < NUM_BIN) {
        local_bin_size[item_rank] = 0;
      }

      if (item_rank == 32) {
        local_local_nnz[0] = 0;
        local_max_row_nnz[0] = 0;
      }

      work_group.barrier();

      auto work_item = work_group.activeItem(0);
      Int32 i = work_item.linearIndex();

      // Guarding against overflow
      if (i < M) {
        row_nnz = row_nnz_view[i];
        ax::doAtomicAdd(&local_local_nnz[0], row_nnz);
        ax::doAtomic<ax::eAtomicOperation::Max, Int32, Int32>(&local_max_row_nnz[0], row_nnz);
        // #pragma unroll
        for (j = 0; j < NUM_BIN; ++j) {
          if (row_nnz <= range[j]) {
            ax::doAtomicAdd(&local_bin_size[j], 1);
            break;
          }
        }
      }

      work_group.barrier();
      if (item_rank < NUM_BIN) {
        ax::doAtomicAdd(bin_size_view[item_rank], local_bin_size[item_rank]);
      }
      if (item_rank == 32) {
        ax::doAtomicAdd(total_nnz_view[0], local_local_nnz[0]);
      }
      if (item_rank == 64) {
        ax::doAtomic<ax::eAtomicOperation::Max, Int32, Int32>(max_row_nnz_view[0], local_max_row_nnz[0]);
      }
    } else {
      for (Int32 b = 0; b < NUM_BIN; ++b) {
        local_bin_size[b] = 0;
      }

      local_local_nnz[0] = 0;
      local_max_row_nnz[0] = 0;

      for (Int32 g = 0; g < work_group.nbActiveItem(); ++g) {
        auto work_item = work_group.activeItem(g);
        Int32 i = work_item.linearIndex() + g;

        // Guarding against overflow
        if (i < M) {
          row_nnz = row_nnz_view[i];
          ax::doAtomicAdd(&local_local_nnz[0], row_nnz);
          ax::doAtomic<ax::eAtomicOperation::Max, Int32, Int32>(&local_max_row_nnz[0], row_nnz);
          // #pragma unroll
          for (j = 0; j < NUM_BIN; ++j) {
            if (row_nnz <= range[j]) {
              ax::doAtomicAdd(&local_bin_size[j], 1);
              break;
            }
          }
        }
      }

      for (Int32 b = 0; b < NUM_BIN; ++b) {
        ax::doAtomicAdd(bin_size_view[b], local_bin_size[b]);
      }
      ax::doAtomicAdd(total_nnz_view[0], local_local_nnz[0]);
      ax::doAtomic<ax::eAtomicOperation::Max, Int32, Int32>(max_row_nnz_view[0], local_max_row_nnz[0]);
    }
  };

  queue->barrier();
}

void ConnectivityMatMul::numericBinningSmall() {
  auto queue = m_meta->run_queues[0].get();
  auto command = makeCommand(queue);
  auto bins_view = ax::viewOut(command, *m_meta->bins);

  command << RUNCOMMAND_LOOP1(iter, m_meta->M) {
    auto [i] = iter();
    bins_view[i] = i;
  };

  queue->barrier();
}

void ConnectivityMatMul::numericBinningSecondPass() {
  const Int32 M = m_meta->M;
  const Int32 group_size = 1024;
  const Int32 nb_groups = div_up(M, group_size);

  auto queue = m_meta->run_queues[0].get();
  auto command = makeCommand(queue);

  auto row_nnz_view = ax::viewIn(command, *m_C.rpt);
  auto bin_offset_view = ax::viewInOut(command, *m_meta->bin_offset);
  auto bin_size_view = ax::viewInOut(command, *m_meta->bin_size);
  auto bins_view = ax::viewOut(command, *m_meta->bins);

  ax::LocalMemory<Int32, NUM_BIN> shared_bin_size(command, NUM_BIN);
  ax::LocalMemory<Int32, NUM_BIN> shared_bin_offset(command, NUM_BIN);

  ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, nb_groups * group_size, nb_groups, group_size);

  command << RUNCOMMAND_LAUNCH(ctx, loop_range, shared_bin_size, shared_bin_offset) {
    auto work_group = ctx.group();
    auto local_bin_size = shared_bin_size.span();
    auto local_bin_offset = shared_bin_offset.span();

    constexpr Int32 range[NUM_BIN] = {31, 255, 511, 1022, 2047, 4095, 8191, INT_MAX};
    Int32 row_nnz, j;

    if (work_group.isDevice()) {
      const Int32 item_rank = work_group.activeWorkItemRankInGroup();
      if (item_rank < NUM_BIN) {
        local_bin_size[item_rank] = 0;
      }

      work_group.barrier();

      auto work_item = work_group.activeItem(0);
      Int32 i = work_item.linearIndex();

      // Guarding against overflow
      if (i < M) {
        row_nnz = row_nnz_view[i];
        //// #pragma unroll
        for (j = 0; j < NUM_BIN; ++j) {
          if (row_nnz <= range[j]) {
            ax::doAtomicAdd(&local_bin_size[j], 1);
            break;
          }
        }
      }

      work_group.barrier();

      if (item_rank < NUM_BIN) {
        local_bin_offset[item_rank] = ax::doAtomicAdd(bin_size_view[item_rank], local_bin_size[item_rank]);
        local_bin_offset[item_rank] += bin_offset_view[item_rank];
        local_bin_size[item_rank] = 0;
      }
      work_group.barrier();
      Int32 index;

      // Guarding against overflow
      if (i < M) {
        // #pragma unroll
        for (j = 0; j < NUM_BIN; ++j) {
          if (row_nnz <= range[j]) {
            index = ax::doAtomicAdd(&local_bin_size[j], 1);
            bins_view[local_bin_offset[j] + index] = i;
            return;
          }
        }
      }
    } else {
      for (Int32 b = 0; b < NUM_BIN; ++b) {
        local_bin_size[b] = 0;
      }

      for (Int32 g = 0; g < work_group.nbActiveItem(); ++g) {
        auto work_item = work_group.activeItem(g);
        Int32 i = work_item.linearIndex() + g;
        Int32 row_nnz, j;

        // Guarding against overflow
        if (i < M) {
          row_nnz = row_nnz_view[i];
          // #pragma unroll
          for (j = 0; j < NUM_BIN; ++j) {
            if (row_nnz <= range[j]) {
              ax::doAtomicAdd(&local_bin_size[j], 1);
              break;
            }
          }
        }

        for (Int32 b = 0; b < NUM_BIN; ++b) {
          local_bin_offset[b] = ax::doAtomicAdd(bin_size_view[b], local_bin_size[b]);
          local_bin_offset[b] += bin_offset_view[b];
          local_bin_size[b] = 0;
        }

        for (Int32 g = 0; g < work_group.nbActiveItem(); ++g) {
          auto work_item = work_group.activeItem(g);
          Int32 i = work_item.linearIndex() + g;

          Int32 index;
          // Guarding against overflow
          if (i < M) {
            // #pragma unroll
            for (j = 0; j < NUM_BIN; ++j) {
              if (row_nnz <= range[j]) {
                index = ax::doAtomicAdd(&local_bin_size[j], 1);
                bins_view[local_bin_offset[j] + index] = i;
                return;
              }
            }
          }
        }
      }
    }
  };

  queue->barrier();
}

} // namespace Connectivix
