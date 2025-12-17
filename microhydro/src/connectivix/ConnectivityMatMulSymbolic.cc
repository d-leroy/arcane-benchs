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

void ConnectivityMatMul::symbolic() {
  if ((*m_meta->bin_size)[5]) {
    symbolicSharedHashTable<8192, 1024>(5);
  }
  Int32 *d_fail_bins, *d_fail_bin_size;
  Int32 fail_bin_size = 0;
  if ((*m_meta->bin_size)[7]) { // shared hash with fail
    ARCANE_FATAL("Unsupported bin size");
    // if(meta.bin_size[7] + 1 <= meta.cub_storage_size/sizeof(Int32)){
    //     d_fail_bins = meta.d_cub_storage;
    //     d_fail_bin_size = meta.d_cub_storage + meta.bin_size[7];
    // }
    // else{ // allocate global memory
    //     CHECK_ERROR(cudaMalloc(&d_fail_bins, (meta.bin_size[7] + 1) *
    //     sizeof(Int32))); d_fail_bin_size = d_fail_bins + meta.bin_size[7];
    // }
    // CHECK_ERROR(cudaMemsetAsync(d_fail_bin_size, 0, sizeof(Int32),
    // meta.stream[7]));
    // CHECK_ERROR(cudaFuncSetAttribute(k_symbolic_max_shared_hash_tb_with_fail,
    //     cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    // k_symbolic_max_shared_hash_tb_with_fail
    //     <<<meta.bin_size[7], 1024, 98304, meta.stream[7]>>>(
    //     A.d_rpt, A.d_col, B.d_rpt, B.d_col,
    //     meta.d_bins + meta.bin_offset[7],
    //     d_fail_bins, d_fail_bin_size,
    //     C.d_rpt);
  }

  if ((*m_meta->bin_size)[6]) {
    ARCANE_FATAL("Unsupported bin size");
    // symbolicSharedHashTable<8192>(*(m_meta->run_queues[6].get()),
    //                               (*m_meta->bin_offset)[6],
    //                               (*m_meta->bin_size)[6]);
    //   k_symbolic_large_shared_hash_tb<<<meta.bin_size[6], 1024, 0,
    //   meta.stream[6]>>>(
    //       A.d_rpt, A.d_col, B.d_rpt, B.d_col,
    //       meta.d_bins + meta.bin_offset[6],
    //       C.d_rpt);
  }

  if ((*m_meta->bin_size)[0]) {
    symbolicPartialWarpSharedHashTable(0);
  }

  if ((*m_meta->bin_size)[7]) {
    ARCANE_FATAL("Unsupported bin size");
    // CHECK_ERROR(cudaMemcpyAsync(&fail_bin_size, d_fail_bin_size,
    // sizeof(Int32), cudaMemcpyDeviceToHost, meta.stream[7]));
    // CHECK_ERROR(cudaStreamSynchronize(meta.stream[7]));
    //     if(fail_bin_size){ // global hash
    //         //printf("inside h_symbolic fail_bin_size %d\n", fail_bin_size);
    //         Int32 max_tsize = *meta.max_row_nnz * SYMBOLIC_SCALE_LARGE;
    //         meta.global_mem_pool_size = fail_bin_size * max_tsize *
    //         sizeof(Int32); CHECK_ERROR(cudaMalloc(&meta.d_global_mem_pool,
    //         meta.global_mem_pool_size)); meta.global_mem_pool_malloced =
    //         true; k_symbolic_global_hash_tb<<<fail_bin_size, 1024, 0,
    //         meta.stream[7]>>>(
    //             A.d_rpt, A.d_col, B.d_rpt, B.d_col,
    //             d_fail_bins,
    //             C.d_rpt, meta.d_global_mem_pool, max_tsize);
    //     }
  }

  if ((*m_meta->bin_size)[4]) {
    symbolicSharedHashTable<4096, 512>(4);
  }

  if ((*m_meta->bin_size)[3]) {
    symbolicSharedHashTable<2048, 256>(3);
  }

  if ((*m_meta->bin_size)[2]) {
    symbolicSharedHashTable<1024, 128>(2);
  }

  if ((*m_meta->bin_size)[1]) {
    symbolicSharedHashTable<512, 64>(1);
  }

  // if (meta.bin_size[7] &&
  //     meta.bin_size[7] + 1 > meta.cub_storage_size / sizeof(Int32)) {
  //   CHECK_ERROR(cudaFree(d_fail_bins));
  // }
}

void ConnectivityMatMul::symbolicPartialWarpSharedHashTable(Int32 bin_index) {
  const ax::RunQueue &queue = *(m_meta->run_queues[bin_index].get());
  const Int32 bin_size = (*m_meta->bin_size)[bin_index];

  const Int32 group_size = PWARP_ROWS * PWARP;
  const Int32 nb_groups = div_up(bin_size, PWARP_ROWS);

  auto command = makeCommand(queue);
  auto arpt_view = ax::viewIn(command, *m_A.rpt);
  auto acol_view = ax::viewIn(command, *m_A.col);
  auto brpt_view = ax::viewIn(command, *m_B.rpt);
  auto bcol_view = ax::viewIn(command, *m_B.col);
  auto bins_view = ax::viewIn(command, *m_meta->bins);
  auto row_nnz_view = ax::viewOut(command, *m_C.rpt);

  ax::LocalMemory<Int32, PWARP_ROWS * PWARP_TSIZE> shared_table(command, PWARP_ROWS * PWARP_TSIZE);
  ax::LocalMemory<Int32, PWARP_ROWS> shared_nnz(command, PWARP_ROWS);

  ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, nb_groups * group_size, nb_groups, group_size);

  command << RUNCOMMAND_LAUNCH(ctx, loop_range, shared_table, shared_nnz) {
    auto work_group = ctx.group();
    // const Int32 group_rank = work_group.groupRank();
    const Int32 group_size = work_group.groupSize();

    auto local_table = shared_table.span();
    auto local_nnz = shared_nnz.span();

    if (work_group.isDevice()) {
      const Int32 item_rank = work_group.activeWorkItemRankInGroup();
      const auto work_item = work_group.activeItem(0);
      const Int32 i = work_item.linearIndex();
      // Thread ID = group rank % number of thread per partial warp.
      const Int32 tid = item_rank & (PWARP - 1);
      // Row ID = index / number of thread per partial warp.
      Int32 rid = i / PWARP;
      const Int32 block_rid = rid & (PWARP_ROWS - 1);

      Int32 j, k;
      for (j = item_rank; j < PWARP_ROWS * PWARP_TSIZE; j += group_size) {
        local_table[j] = -1;
      }
      if (item_rank < PWARP_ROWS) {
        local_nnz[item_rank] = 0;
      }
      if (rid >= bin_size) {
        return;
      }

      work_group.barrier();

      Int32 *table = &local_table[block_rid * PWARP_TSIZE];

      rid = bins_view[rid];
      Int32 acol, bcol;
      Int32 hash, old;
      for (j = arpt_view[rid] + tid; j < arpt_view[rid + 1]; j += PWARP) { // pwarp per row, thread per a item, thread per b row
        acol = acol_view[j];
        for (k = brpt_view[acol]; k < brpt_view[acol + 1]; ++k) { // thread per b row
          bcol = bcol_view[k];
          hash = (bcol * HASH_SCALE) & (PWARP_TSIZE - 1);
          while (1) {
            old = ax::doAtomicCAS(&table[hash], -1, bcol);
            if (old == -1) {
              ax::doAtomicAdd(&local_nnz[block_rid], 1);
              break;
            } else if (old == bcol) {
              break;
            } else {
              hash = (hash + 1) & (PWARP_TSIZE - 1);
            }
          }
        }
      }

      work_group.barrier();

      if (tid == 0) {
        row_nnz_view[rid] = local_nnz[block_rid];
      }
    } else {
      const Int32 nb_items = work_group.nbActiveItem();

      // std::cout << "Handling " << nb_items << " items" << std::endl;
      // std::cout << "NNZ: " << arpt_view[m_meta->M] << std::endl;

      for (Int32 item_rank = 0; item_rank < nb_items; ++item_rank) {
        for (Int32 j = item_rank; j < PWARP_ROWS * PWARP_TSIZE; j += group_size) {
          local_table[j] = -1;
        }
        if (item_rank < PWARP_ROWS) {
          local_nnz[item_rank] = 0;
        }
      }

      work_group.barrier();

      for (Int32 item_rank = 0; item_rank < nb_items; ++item_rank) {
        const auto work_item = work_group.activeItem(item_rank);
        // Linear thread index
        const Int32 i = work_item.linearIndex();
        // Each row is processed by PWARP threads
        Int32 rid = i / PWARP;

        // Guard against overflow
        if (rid >= bin_size) {
          continue;
        };

        const Int32 tid = item_rank & (PWARP - 1);
        const Int32 block_rid = rid & (PWARP_ROWS - 1);
        const Int32 table_offset = block_rid * PWARP_TSIZE;

        rid = bins_view[rid];
        Int32 j, k, acol, bcol, hash, old;

        if (arpt_view[rid] + tid < arpt_view[rid + 1]) {
          // std::cout << std::endl;
          // std::cout << "================= Row ID: " << rid << " [" << tid << "] =================" << std::endl;
        }

        for (j = arpt_view[rid] + tid; j < arpt_view[rid + 1]; j += PWARP) { // pwarp per row, thread per a item, thread per b row
          // std::cout << "  column at index " << j << std::endl;
          acol = acol_view[j];
          // std::cout << "  A(" << rid << "," << acol << ")" << std::endl;
          for (k = brpt_view[acol]; k < brpt_view[acol + 1]; ++k) { // thread per b row
            bcol = bcol_view[k];
            // std::cout << "  B(" << acol << "," << bcol << ")" << std::endl;
            hash = (bcol * HASH_SCALE) & (PWARP_TSIZE - 1);
            // std::cout << "  hash = " << hash << std::endl;
            while (1) {
              old = ax::doAtomicCAS(&local_table[table_offset + hash], -1, bcol);
              if (old == -1) {
                // std::cout << "  => Product A(" << rid << "," << acol << ") x B(" << acol << "," << bcol << ") for C(" << rid << "," << bcol << ")" << std::endl;
                ax::doAtomicAdd(&local_nnz[block_rid], 1);
                break;
              } else if (old == bcol) {
                // std::cout << "  => Product A(" << rid << "," << acol << ") x B(" << acol << "," << bcol << ") for C(" << rid << "," << bcol << ") (already registered)" << std::endl;
                break;
              } else {
                hash = (hash + 1) & (PWARP_TSIZE - 1);
              }
            }
          }
        }
      }

      work_group.barrier();

      for (Int32 item_rank = 0; item_rank < nb_items; ++item_rank) {
        const auto work_item = work_group.activeItem(item_rank);
        const Int32 i = work_item.linearIndex();
        const Int32 tid = item_rank & (PWARP - 1);
        Int32 rid = i / PWARP;

        if (rid >= bin_size) {
          continue;
        }

        const Int32 block_rid = rid & (PWARP_ROWS - 1);
        rid = bins_view[rid];

        if (tid == 0) {
          row_nnz_view[rid] = local_nnz[block_rid];
        }
      }
    }
  };
}

} // namespace Connectivix