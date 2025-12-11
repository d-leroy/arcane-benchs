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

void ConnectivityMatMul::numeric() {
  if ((*m_meta->bin_size)[6]) {
    // CHECK_ERROR(cudaFuncSetAttribute(
    //     k_numeric_max_shared_hash_tb_half_occu,
    //     cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    // k_numeric_max_shared_hash_tb_half_occu<<<(*m_meta->bin_size)[6], 1024,
    // 98304,
    //                                          meta.stream[6]>>>(
    //     A.d_rpt, A.d_col, B.d_rpt, B.d_col, meta.d_bins + meta.bin_offset[6],
    //     C.d_rpt, C.d_col);
  }

  if ((*m_meta->bin_size)[7]) { // global bin
    // Int32 max_tsize = *meta.max_row_nnz * NUMERIC_SCALE_LARGE;
    // size_t global_size =
    //     (*m_meta->bin_size)[7] * max_tsize * (sizeof(Int32) +
    //     sizeof(mdouble));
    // if (meta.global_mem_pool_malloced) {
    //   if (global_size <= meta.global_mem_pool_size) {
    //     // do nothing
    //   } else {
    //     CHECK_ERROR(cudaFree(meta.d_global_mem_pool));
    //     CHECK_ERROR(cudaMalloc(&meta.d_global_mem_pool, global_size));
    //   }
    // } else {
    //   CHECK_ERROR(cudaMalloc(&meta.d_global_mem_pool, global_size));
    //   meta.global_mem_pool_size = global_size;
    //   meta.global_mem_pool_malloced = true;
    // }
    // k_numeric_global_hash_tb_full_occu<<<(*m_meta->bin_size)[7], 1024, 0,
    //                                      meta.stream[7]>>>(
    //     A.d_rpt, A.d_col, B.d_rpt, B.d_col, meta.d_bins + meta.bin_offset[7],
    //     max_tsize, meta.d_global_mem_pool, C.d_rpt, C.d_col);
  }

  if ((*m_meta->bin_size)[5]) {
    numericSharedHashTable<4096, 1024>(5);
  }

  if ((*m_meta->bin_size)[0]) {
    numericPartialWarpSharedHashTable(0);
  }

  if ((*m_meta->bin_size)[4]) {
    numericSharedHashTable<2048, 512>(4);
  }

  if ((*m_meta->bin_size)[3]) {
    numericSharedHashTable<1024, 256>(3);
  }

  if ((*m_meta->bin_size)[2]) {
    numericSharedHashTable<512, 128>(2);
  }
  if ((*m_meta->bin_size)[1]) {
    numericSharedHashTable<256, 64>(1);
  }

  // if (meta.global_mem_pool_malloced) {
  //   CHECK_ERROR(cudaFree(meta.d_global_mem_pool));
  // }
}

void ConnectivityMatMul::numericPartialWarpSharedHashTable(Int32 bin_index) {
  const ax::RunQueue &queue = *(m_meta->run_queues[bin_index].get());
  const Int32 bin_size = (*m_meta->bin_size)[bin_index];

  constexpr Int32 group_size = NUMERIC_PWARP_ROWS * NUMERIC_PWARP;
  constexpr Int32 tsize = NUMERIC_PWARP_TSIZE - 1;
  const Int32 nb_groups = div_up(bin_size, NUMERIC_PWARP_ROWS);

  auto command = makeCommand(queue);
  auto arpt_view = ax::viewIn(command, *m_A.rpt);
  auto acol_view = ax::viewIn(command, *m_A.col);
  auto brpt_view = ax::viewIn(command, *m_B.rpt);
  auto bcol_view = ax::viewIn(command, *m_B.col);
  auto bins_view = ax::viewIn(command, *m_meta->bins);
  auto crpt_view = ax::viewIn(command, *m_C.rpt);
  auto ccol_view = ax::viewInOut(command, *m_C.col);

  ax::LocalMemory<Int32, NUMERIC_PWARP_ROWS * tsize> shared_hash(command, NUMERIC_PWARP_ROWS * tsize);
  ax::LocalMemory<Int32, NUMERIC_PWARP_ROWS * tsize> shared_col(command, NUMERIC_PWARP_ROWS * tsize);
  ax::LocalMemory<Int32, NUMERIC_PWARP_ROWS> shared_offset(command, NUMERIC_PWARP_ROWS);

  ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, nb_groups * group_size, nb_groups, group_size);

  // std::cout << std::endl;
  // std::cout << "===========================================" << std::endl;
  // std::cout << "================= NUMERIC =================" << std::endl;
  // std::cout << "===========================================" << std::endl;
  // std::cout << std::endl;

  command << RUNCOMMAND_LAUNCH(ctx, loop_range, shared_hash, shared_col, shared_offset) {
    auto work_group = ctx.group();
    const Int32 group_rank = work_group.groupRank();
    const Int32 group_size = work_group.groupSize();

    // std::cout << "================= GROUP RANK: " << group_rank << " =================" << std::endl;

    auto local_hash = shared_hash.span();
    auto local_col = shared_col.span();
    auto local_offset = shared_offset.span();

    if constexpr (work_group.isDevice()) {
      const Int32 item_rank = work_group.activeWorkItemRankInGroup();
      const auto work_item = work_group.activeItem(0);
      const Int32 i = work_item.linearIndex();
      const Int32 tid = item_rank % NUMERIC_PWARP;
      Int32 rid = i / NUMERIC_PWARP;
      const Int32 block_rid = rid % NUMERIC_PWARP_ROWS;
      const Int32 table_offset = block_rid * tsize;
      Int32 j, k;
      for (j = item_rank; j < NUMERIC_PWARP_ROWS * tsize; j += group_size) {
        local_hash[j] = -1;
      }
      if (item_rank < NUMERIC_PWARP_ROWS) {
        local_offset[item_rank] = 0;
      }
      if (rid >= bin_size) {
        return;
      }
      work_group.barrier();

      rid = bins_view[rid];
      Int32 acol, bcol, hash, old;
      for (j = arpt_view[rid] + tid; j < arpt_view[rid + 1]; j += NUMERIC_PWARP) { // pwarp per row, thread per a item, thread per
                                                                                   // b row
        acol = acol_view[j];
        for (k = brpt_view[acol]; k < brpt_view[acol + 1]; ++k) { // thread per b row
          bcol = bcol_view[k];
          hash = (bcol * HASH_SCALE) % tsize;
          while (1) {
            old = atomicCAS(&local_hash[table_offset + hash], -1, bcol);
            if (old == -1 || old == bcol) {
              break;
            } else {
              hash = (hash + 1) < tsize ? hash + 1 : 0;
            }
          }
        }
      }
      work_group.barrier();

      const Int32 c_offset = crpt_view[rid];
      const Int32 row_nnz = crpt_view[rid + 1] - crpt_view[rid];
      Int32 offset;
      bool valid;
      // #pragma unroll
      for (j = 0; j < tsize; j += NUMERIC_PWARP) {
        offset = j + tid;
        valid = offset < tsize;
        if (valid) {
          acol = local_hash[table_offset + offset];
          if (acol != -1) {
            offset = ax::doAtomicAdd(&local_offset[block_rid], 1);
          }
        }
        work_group.barrier();
        if (valid && acol != -1) {
          local_col[table_offset + offset] = acol;
        }
      }

      work_group.barrier();

      // count sort the result
      for (j = tid; j < row_nnz; j += NUMERIC_PWARP) {
        acol = local_col[table_offset + j];
        offset = 0;
        for (k = 0; k < row_nnz; ++k) {
          offset += (unsigned int)(local_col[table_offset + k] - acol) >> /*31*/ 31;
        }
        ccol_view[c_offset + offset] = local_col[table_offset + j];
      }
    } else {
      const Int32 nb_active_items = work_group.nbActiveItem();

      for (Int32 item_rank = 0; item_rank < nb_active_items; ++item_rank) {
        const auto work_item = work_group.activeItem(item_rank);
        Int32 j;

        for (j = item_rank; j < NUMERIC_PWARP_ROWS * tsize; j += group_size) {
          local_hash[j] = -1;
        }
        if (item_rank < NUMERIC_PWARP_ROWS) {
          local_offset[item_rank] = 0;
        }
      }

      work_group.barrier();

      for (Int32 item_rank = 0; item_rank < nb_active_items; ++item_rank) {
        const auto work_item = work_group.activeItem(item_rank);
        const Int32 i = work_item.linearIndex();
        Int32 rid = i / NUMERIC_PWARP;

        if (rid >= bin_size) {
          continue;
        }

        const Int32 tid = item_rank % NUMERIC_PWARP;
        const Int32 block_rid = rid % NUMERIC_PWARP_ROWS;
        const Int32 table_offset = block_rid * tsize;

        rid = bins_view[rid];
        Int32 j, k, acol, bcol, hash, old;

        if (arpt_view[rid] + tid < arpt_view[rid + 1]) {
          // std::cout << std::endl;
          // std::cout << "================= Row ID: " << rid << " [" << tid << "] =================" << std::endl;
        }

        for (j = arpt_view[rid] + tid; j < arpt_view[rid + 1]; j += NUMERIC_PWARP) { // pwarp per row, thread per a item, thread per b row
          // std::cout << "  column at index " << j << std::endl;
          acol = acol_view[j];
          // std::cout << "  A(" << rid << "," << acol << ")" << std::endl;
          for (k = brpt_view[acol]; k < brpt_view[acol + 1]; ++k) { // thread per b row
            bcol = bcol_view[k];
            // std::cout << "  B(" << acol << "," << bcol << ")" << std::endl;
            hash = (bcol * HASH_SCALE) % tsize;
            // std::cout << "  hash = " << hash << std::endl;
            while (1) {
              old = atomicCAS(&local_hash[table_offset + hash], -1, bcol);
              if (old == -1 || old == bcol) {
                // std::cout << "  => Product A(" << rid << "," << acol << ") x B(" << acol << "," << bcol << ") for C(" << rid << "," << bcol << ")" << std::endl;
                // std::cout << "Hash table at " << table_offset << ": ";
                for (Int32 x = 0; x < tsize; ++x) {
                  // std::cout << local_hash[table_offset + x] << " ";
                }
                // std::cout << std::endl;
                break;
              } else {
                hash = (hash + 1) < tsize ? hash + 1 : 0;
              }
            }
          }
        }
      }

      work_group.barrier();

      for (Int32 item_rank = 0; item_rank < nb_active_items; ++item_rank) {
        const auto work_item = work_group.activeItem(item_rank);
        const Int32 i = work_item.linearIndex();
        Int32 rid = i / NUMERIC_PWARP;

        if (rid >= bin_size) {
          continue;
        }

        const Int32 block_rid = rid % NUMERIC_PWARP_ROWS;
        const Int32 tid = item_rank % NUMERIC_PWARP;
        const Int32 table_offset = block_rid * tsize;

        rid = bins_view[rid];
        // std::cout << "================= Row ID: " << rid << " [" << tid << "] =================" << std::endl;

        // std::cout << "Hash table at " << table_offset << ": ";
        for (Int32 x = 0; x < tsize; ++x) {
          // std::cout << local_hash[table_offset + x] << " ";
        }
        // std::cout << std::endl;

        Int32 j, k, acol, offset;
        bool valid;

        if constexpr (work_group.isDevice()) {
          // #pragma unroll
          for (j = 0; j < tsize; j += NUMERIC_PWARP) {
            offset = tid + j;
            valid = offset < tsize;
            if (valid) {
              acol = local_hash[table_offset + offset];
              // If there's a column in the hash table at 'offset'
              if (acol != -1) {
                offset = ax::doAtomicAdd(&local_offset[block_rid], 1);
              }
            }
            work_group.barrier();
            // Since we wait for all threads before writing to tid + j,
            // threads don't overwrite data before it is read by another thread.
            if (valid && acol != -1) {
              local_col[table_offset + offset] = acol;
            }
          }
        } else {
          for (j = 0; j < tsize; j += NUMERIC_PWARP) {
            offset = tid + j;
            // std::cout << "  Offset " << offset << std::endl;
            valid = offset < tsize;
            if (valid) {
              acol = local_hash[table_offset + offset];
              // If there's a column in the hash table at 'offset'
              if (acol != -1) {
                offset = ax::doAtomicAdd(&local_offset[block_rid], 1);
                // std::cout << "  Offset in columns: " << offset << std::endl;
                // std::cout << "  Column:            " << acol << std::endl;
              }
            }

            if (valid && acol != -1) {
              local_col[table_offset + offset] = acol;
            }
          }
        }

        // std::cout << "Hash table at " << table_offset << ": ";
        for (Int32 x = 0; x < tsize; ++x) {
          // std::cout << local_hash[table_offset + x] << " ";
        }
        // std::cout << std::endl;

        // std::cout << "Col table at " << table_offset << ": ";
        for (Int32 x = 0; x < tsize; ++x) {
          // std::cout << local_col[table_offset + x] << " ";
        }
        // std::cout << std::endl;
      }

      work_group.barrier();

      for (Int32 item_rank = 0; item_rank < nb_active_items; ++item_rank) {
        const auto work_item = work_group.activeItem(item_rank);
        const Int32 i = work_item.linearIndex();
        Int32 rid = i / NUMERIC_PWARP;

        if (rid >= bin_size) {
          continue;
        }

        const Int32 tid = item_rank % NUMERIC_PWARP;
        const Int32 block_rid = rid % NUMERIC_PWARP_ROWS;
        const Int32 table_offset = block_rid * tsize;

        rid = bins_view[rid];
        Int32 j, k, acol, offset;

        // std::cout << "================= Row ID: " << rid << " [" << tid << "] =================" << std::endl;

        const Int32 c_offset = crpt_view[rid];
        const Int32 row_nnz = crpt_view[rid + 1] - crpt_view[rid];

        // count sort the result
        for (j = tid; j < row_nnz; j += NUMERIC_PWARP) {
          acol = local_col[table_offset + j];
          offset = 0;
          for (k = 0; k < row_nnz; ++k) {
            offset += (unsigned int)(local_col[table_offset + k] - acol) >> 31;
          }
          ccol_view[c_offset + offset] = local_col[table_offset + j];
        }
      }
    }
  };
}

} // namespace Connectivix