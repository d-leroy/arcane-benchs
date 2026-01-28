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
#include "arccore/common/accelerator/DeviceInfo.h"
#include "define.h"
#include <algorithm>

namespace ax = Arcane::Accelerator;

namespace Connectivix {

void ConnectivityMatMul::symbolic() {
  if ((*m_meta->bin_size)[5]) {
    symbolicSharedHashTable<8192, 1024>(5);
  }
  Int32 *d_fail_bins, *d_fail_bin_size;
  Int32 fail_bin_size = 0;
  if ((*m_meta->bin_size)[7]) { // shared hash with fail
    ARCANE_FATAL("Unsupported bin size: 7");
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
    ARCANE_FATAL("Unsupported bin size: 6");
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
    ARCANE_FATAL("Unsupported bin size: 7");
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

  const auto &device_info = m_runner.deviceInfo();

  // 1024
  // const Int32 maxThreadsPerBlock = device_info.maxThreadsPerBlock();
  const Int32 maxThreadsPerBlock = 1024;
  // 1536
  // const Int32 maxThreadsPerSM = device_info.maxThreadsPerMultiProcessor();
  const Int32 maxThreadsPerSM = 1536;

  // 2 => threads per SM divided by threads per block, rounded up
  const Int32 nbBlocks = div_up(maxThreadsPerSM, maxThreadsPerBlock);
  // 768 => threads per SM divided by number of blocks, rounded down to the nearest multiple of 32 (warp size)
  const Int32 nbThreadsPerBlock = (div_up(maxThreadsPerSM, nbBlocks) / 32) * 32;

  // 49152
  const Int32 shmemPerBlock = device_info.sharedMemoryPerBlock();
  // 102400
  const Int32 shmemPerSM = device_info.sharedMemoryPerMultiprocessor();
  // 49152 (TODO: take optin into account?)
  const Int32 maxShmemPerBlock = std::min(shmemPerBlock, shmemPerSM / nbBlocks);

  // 192 => each partial warp (4 threads) handles one row, so each block can handle at most (threads per block) / 4 rows
  const Int32 rows_per_block_comp = nbThreadsPerBlock / PWARP;
  // 64 => each row has its hash table, so there are (max shared memory per block) / (rows per block) / 4 entries of 4 bytes available per row
  const Int32 rows_per_block_shmem = maxShmemPerBlock / (4 * (PWARP_TSIZE + 1));

  const Int32 rows_per_block = PWARP_ROWS; // std::min(rows_per_block_shmem, rows_per_block_comp);

  // How many threads in a block = rows in a block * threads per partial warp (= 4)
  //
  const Int32 group_size = rows_per_block * PWARP;
  //
  const Int32 nb_groups = div_up(bin_size, rows_per_block);

  auto command = makeCommand(queue);
  auto arpt_view = ax::viewIn(command, *m_A.rpt);
  auto acol_view = ax::viewIn(command, *m_A.col);
  auto brpt_view = ax::viewIn(command, *m_B.rpt);
  auto bcol_view = ax::viewIn(command, *m_B.col);
  auto bins_view = ax::viewIn(command, *m_meta->bins);
  auto row_nnz_view = ax::viewOut(command, *m_C.rpt);

  ax::LocalMemory<Int32> shared_table(command, rows_per_block * PWARP_TSIZE);
  ax::LocalMemory<Int32> shared_nnz(command, rows_per_block);

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
        }

        for (j = arpt_view[rid] + tid; j < arpt_view[rid + 1]; j += PWARP) { // pwarp per row, thread per a item, thread per b row
          acol = acol_view[j];
          for (k = brpt_view[acol]; k < brpt_view[acol + 1]; ++k) { // thread per b row
            bcol = bcol_view[k];
            hash = (bcol * HASH_SCALE) & (PWARP_TSIZE - 1);
            while (1) {
              old = ax::doAtomicCAS(&local_table[table_offset + hash], -1, bcol);
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