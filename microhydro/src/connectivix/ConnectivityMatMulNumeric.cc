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
    ARCANE_FATAL("Unsupported bin size");
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
    ARCANE_FATAL("Unsupported bin size");
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

  if ((*m_meta->bin_size)[0]) {
    numericPartialWarpSharedHashTable(0);
  }

  if ((*m_meta->bin_size)[5]) {
    numericSharedHashTable<4096, 1024>(5);
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
    numericSharedHashTable<512, 64>(1);
  }

  // if (meta.global_mem_pool_malloced) {
  //   CHECK_ERROR(cudaFree(meta.d_global_mem_pool));
  // }
}

void ConnectivityMatMul::numericPartialWarpSharedHashTable(Int32 bin_index) {
  const ax::RunQueue &queue = *(m_meta->run_queues[bin_index].get());
  const Int32 bin_size = (*m_meta->bin_size)[bin_index];

  constexpr Int32 group_size = NUMERIC_PWARP_ROWS * NUMERIC_PWARP;
  const Int32 nb_groups = div_up(bin_size, NUMERIC_PWARP_ROWS);

  auto command = makeCommand(queue);
  auto arpt_view = ax::viewIn(command, *m_A.rpt);
  auto acol_view = ax::viewIn(command, *m_A.col);
  auto brpt_view = ax::viewIn(command, *m_B.rpt);
  auto bcol_view = ax::viewIn(command, *m_B.col);
  auto bins_view = ax::viewIn(command, *m_meta->bins);
  auto crpt_view = ax::viewIn(command, *m_C.rpt);
  auto ccol_view = ax::viewInOut(command, *m_C.col);

  ax::LocalMemory<Int32, NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE> local_hash(command, NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE);
  ax::LocalMemory<Int32, NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE> local_col(command, NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE);
  ax::LocalMemory<Int32, NUMERIC_PWARP_ROWS> local_offset(command, NUMERIC_PWARP_ROWS);

  ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, nb_groups * group_size, nb_groups, group_size);

  std::cout << "Beginning numeric phase in bin: " << bin_index << std::endl;
  // For each row, hash table is of size NUMERIC_PWARP_TSIZE
  std::cout << "Hash table size = " << NUMERIC_PWARP_TSIZE << std::endl;

  command << RUNCOMMAND_LAUNCH(ctx, loop_range, local_hash, local_col, local_offset) {
    auto work_group = ctx.group();
    const Int32 group_rank = work_group.groupRank();
    const Int32 group_size = work_group.groupSize();

    auto shared_hash = local_hash.span();
    auto shared_col = local_col.span();
    auto shared_offset = local_offset.span();

    if (work_group.isDevice()) {
      const Int32 item_rank = work_group.activeWorkItemRankInGroup();
      const auto work_item = work_group.activeItem(0);
      const Int32 i = work_item.linearIndex();
      const Int32 tid = item_rank % NUMERIC_PWARP;                // thread id in the partial warp (NUMERIC_PWARP threads in a partial warp)
      Int32 rid = i / NUMERIC_PWARP;                              // global row id (each row is computed by NUMERIC_PWARP threads)
      const Int32 group_rid = rid % NUMERIC_PWARP_ROWS;           // group row id (each thread group handles NUMERIC_PWARP_ROWS rows)
      const Int32 table_offset = group_rid * NUMERIC_PWARP_TSIZE; // offset of the row's hash table in the group-shared hash table (row hash table are of size NUMERIC_PWARP_TSIZE)
      Int32 j, k;
      // Initializing the group-shared hash table.
      // Each row in the group (NUMERIC_PWARP_ROWS rows in a group) is computed by NUMERIC_PWARP threads, and uses a hash table of size NUMERIC_PWARP_TSIZE.
      for (j = item_rank; j < NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE; j += group_size) {
        shared_hash[j] = -1;
      }
      // Initializing the offsets
      if (item_rank < NUMERIC_PWARP_ROWS) {
        shared_offset[item_rank] = 0;
      }
      if (rid >= bin_size) {
        return;
      }
      work_group.barrier();

      // Retrieve the actual row id to be computed from the bin
      rid = bins_view[rid];
      Int32 acol, bcol, hash, old;
      // Each thread loops on the columns of A:
      // - thread 0 loops on 0, 0 + NUMERIC_PWARP, 0 + 2 * NUMERIC_PWARP, ...
      // - thread 1 loops on 1, 1 + NUMERIC_PWARP, 1 + 2 * NUMERIC_PWARP, ...
      for (j = arpt_view[rid] + tid; j < arpt_view[rid + 1]; j += NUMERIC_PWARP) {
        acol = acol_view[j];
        // Looping on the columns of the row in B corresponding to the current column in A
        for (k = brpt_view[acol]; k < brpt_view[acol + 1]; ++k) {
          bcol = bcol_view[k];
          // Computing a base hash value (i.e., an index in the hash table) for the current column index in B.
          hash = (bcol * HASH_SCALE) % NUMERIC_PWARP_TSIZE;
          while (1) {
            // Increment the hash value as long as there is a collision in the hash table.
            old = ax::doAtomicCAS(&shared_hash[table_offset + hash], -1, bcol);
            if (old == -1 || old == bcol) {
              break;
            } else {
              hash = (hash + 1) % NUMERIC_PWARP_TSIZE;
            }
          }
        }
      }
      work_group.barrier();

      // Index of first column on the row in the row array.
      const Int32 c_offset = crpt_view[rid];
      // Number of non-zero values in the row (i.e., number of columns in the row)
      const Int32 row_nnz = crpt_view[rid + 1] - crpt_view[rid];
      Int32 offset;
      bool valid;
      // #pragma unroll
      // Looping on the hash table.
      for (j = 0; j < NUMERIC_PWARP_TSIZE; j += NUMERIC_PWARP) {
        // Offset in the hash table, depending on j and thread id in the partial warp.
        // We initialize j at 0 and not tid to avoid a deadlock in the nested barrier.
        offset = j + tid;
        valid = offset < NUMERIC_PWARP_TSIZE;
        if (valid) {
          // Retrieving potential column at offset in hash table.
          acol = shared_hash[table_offset + offset];
          if (acol != -1) {
            // A column was found, resulting in actual nnz value.
            // We retrieve the offset in the column array of the resulting matrix at which the thread will write its computed column indices,
            // and increment the offset of the next column to be written there, for the row in this thread's thread group.
            offset = ax::doAtomicAdd(&shared_offset[group_rid], 1);
          }
        }
        // Beware of the potential deadlock here.
        work_group.barrier();
        // By now, each thread in each partial warp of the thread group has finished computing an offset for their current column.
        if (valid && acol != -1) {
          // Writing the column index at the computed offset.
          shared_col[table_offset + offset] = acol;
        }
      }

      work_group.barrier();

      // count sort the result
      for (j = tid; j < row_nnz; j += NUMERIC_PWARP) {
        acol = shared_col[table_offset + j];
        offset = 0;
        for (k = 0; k < row_nnz; ++k) {
          // Branchless comparison: increment offset only if shared_col[k] < target
          offset += (unsigned int)(shared_col[table_offset + k] - acol) >> 31;
        }
        ccol_view[c_offset + offset] = shared_col[table_offset + j];
      }
    } else {
      const Int32 nb_active_items = work_group.nbActiveItem();

      for (Int32 item_rank = 0; item_rank < nb_active_items; ++item_rank) {
        const auto work_item = work_group.activeItem(item_rank);
        Int32 j;

        for (j = item_rank; j < NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE; j += group_size) {
          shared_hash[j] = -1;
        }
        if (item_rank < NUMERIC_PWARP_ROWS) {
          shared_offset[item_rank] = 0;
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
        const Int32 table_offset = block_rid * NUMERIC_PWARP_TSIZE;

        rid = bins_view[rid];
        Int32 j, k, acol, bcol, hash, old;

        for (j = arpt_view[rid] + tid; j < arpt_view[rid + 1]; j += NUMERIC_PWARP) { // pwarp per row, thread per a item, thread per b row
          acol = acol_view[j];
          for (k = brpt_view[acol]; k < brpt_view[acol + 1]; ++k) { // thread per b row
            bcol = bcol_view[k];
            hash = (bcol * HASH_SCALE) % NUMERIC_PWARP_TSIZE;
            while (1) {
              old = ax::doAtomicCAS(&shared_hash[table_offset + hash], -1, bcol);
              if (old == -1 || old == bcol) {
                for (Int32 x = 0; x < NUMERIC_PWARP_TSIZE; ++x) {
                }
                break;
              } else {
                hash = (hash + 1) < NUMERIC_PWARP_TSIZE ? hash + 1 : 0;
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
        const Int32 table_offset = block_rid * NUMERIC_PWARP_TSIZE;

        rid = bins_view[rid];

        for (Int32 x = 0; x < NUMERIC_PWARP_TSIZE; ++x) {
        }

        Int32 j, acol, offset;
        bool valid;

        if (work_group.isDevice()) {
          // #pragma unroll
          for (j = 0; j < NUMERIC_PWARP_TSIZE; j += NUMERIC_PWARP) {
            offset = tid + j;
            valid = offset < NUMERIC_PWARP_TSIZE;
            if (valid) {
              acol = shared_hash[table_offset + offset];
              // If there's a column in the hash table at 'offset'
              if (acol != -1) {
                offset = ax::doAtomicAdd(&shared_offset[block_rid], 1);
              }
            }
            work_group.barrier();
            // Since we wait for all threads before writing to tid + j,
            // threads don't overwrite data before it is read by another thread.
            if (valid && acol != -1) {
              shared_col[table_offset + offset] = acol;
            }
          }
        } else {
          for (j = 0; j < NUMERIC_PWARP_TSIZE; j += NUMERIC_PWARP) {
            offset = tid + j;
            valid = offset < NUMERIC_PWARP_TSIZE;
            if (valid) {
              acol = shared_hash[table_offset + offset];
              // If there's a column in the hash table at 'offset'
              if (acol != -1) {
                offset = ax::doAtomicAdd(&shared_offset[block_rid], 1);
              }
            }

            if (valid && acol != -1) {
              shared_col[table_offset + offset] = acol;
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

        const Int32 tid = item_rank % NUMERIC_PWARP;
        const Int32 block_rid = rid % NUMERIC_PWARP_ROWS;
        const Int32 table_offset = block_rid * NUMERIC_PWARP_TSIZE;

        rid = bins_view[rid];
        Int32 j, k, acol, offset;

        const Int32 c_offset = crpt_view[rid];
        const Int32 row_nnz = crpt_view[rid + 1] - crpt_view[rid];

        // count sort the result
        for (j = tid; j < row_nnz; j += NUMERIC_PWARP) {
          acol = shared_col[table_offset + j];
          offset = 0;
          for (k = 0; k < row_nnz; ++k) {
            offset += (unsigned int)(shared_col[table_offset + k] - acol) >> 31;
          }
          ccol_view[c_offset + offset] = shared_col[table_offset + j];
        }
      }
    }
  };
}

} // namespace Connectivix