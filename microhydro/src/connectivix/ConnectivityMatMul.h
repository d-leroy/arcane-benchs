#ifndef CONNECTIVIX_CONNECTIVITY_MAT_MUL_H
#define CONNECTIVIX_CONNECTIVITY_MAT_MUL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "CSR.h"
#include "Metadata.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/Atomic.h"
#include "arcane/accelerator/LocalMemory.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLaunch.h"
#include "define.h"
#include <cstddef>
#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;
using namespace Arcane;

namespace Arcane::Accelerator {
ARCCORE_HOST_DEVICE
inline Int32 doAtomicCAS(Int32 *addr, Int32 expected, Int32 desired) {
#if defined(__CUDA_ARCH__)
  // device-side: directly call CUDA atomicCAS variants
  // Note: atomicCAS supports int/unsigned/unsigned long long on many devices;
  // for other types (e.g. 64-bit on older compute capabilities) adapt as
  // required.
  return atomicCAS(addr, expected, desired);
#elif defined(__CUDACC__) && !defined(__CUDA_ARCH__)
  // Compiling host code with nvcc (host side compilation)
  // Use std::atomic_ref on host as well.
  std::atomic_ref<Int32> a(*addr);
  Int32 expected_copy = expected;
  a.compare_exchange_strong(expected_copy, desired);
  return expected_copy;
#else
  // Pure host (non-CUDA) compilation
  std::atomic_ref<Int32> a(*addr);
  Int32 expected_copy = expected;
  a.compare_exchange_strong(expected_copy, desired);
  return expected_copy;
#endif
}
} // namespace Arcane::Accelerator

namespace ax = Arcane::Accelerator;

namespace Connectivix {

class ConnectivityMatMul {
public:
  ConnectivityMatMul(const CSR &A, const CSR &B, CSR &C, ax::Runner &runner) : m_A(A), m_B(B), m_C(C), m_runner(runner) {
    m_meta = new Metadata(C, runner);
  };
  ~ConnectivityMatMul() {
    m_meta->release();
  };

public:
  void doMatMul();

public:
  void setup();
  void symbolicBinning();
  void symbolic();
  void numericBinning();
  void numeric();

  void symbolicBinningSmall();
  void symbolicBinningFirstPass();
  void symbolicBinningSecondPass();

  void symbolicPartialWarpSharedHashTable(Int32 bin_index);
  template <Int32 SH_ROW, Int32 GROUP_SIZE> void symbolicSharedHashTable(Int32 bin_index);

  void numericBinningSmall();
  void numericBinningFirstPass();
  void numericBinningSecondPass();

  void numericPartialWarpSharedHashTable(Int32 bin_index);
  template <Int32 SH_ROW, Int32 GROUP_SIZE> void numericSharedHashTable(Int32 bin_index);

private:
  ax::Runner &m_runner;
  Metadata *m_meta;
  const CSR &m_A;
  const CSR &m_B;
  CSR &m_C;
};

template <Int32 SH_ROW, Int32 GROUP_SIZE> void ConnectivityMatMul::symbolicSharedHashTable(Int32 bin_index) {
  const ax::RunQueue &queue = m_meta->get_run_queue(bin_index);
  const Int32 bin_offset = m_meta->get_bin_offset(bin_index);
  const Int32 bin_size = m_meta->get_bin_size(bin_index);

  auto command = makeCommand(queue);
  auto arpt_view = ax::viewIn(command, *m_A.rpt);
  auto acol_view = ax::viewIn(command, *m_A.col);
  auto brpt_view = ax::viewIn(command, *m_B.rpt);
  auto bcol_view = ax::viewIn(command, *m_B.col);
  auto bins_view = ax::viewIn(command, *m_meta->bins);
  auto row_nnz_view = ax::viewInOut(command, *m_C.rpt);

  ax::LocalMemory<Int32, SH_ROW> shared_table(command, SH_ROW);
  ax::LocalMemory<Int32, 1> shared_nnz(command, 1);

  ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, bin_size * GROUP_SIZE, bin_size, GROUP_SIZE);

  command << RUNCOMMAND_LAUNCH(ctx, loop_range, shared_table, shared_nnz) {
    auto work_group = ctx.group();
    const Int32 group_rank = work_group.groupRank();
    const Int32 group_size = work_group.groupSize();

    auto local_table = shared_table.span();
    auto local_nnz = shared_nnz.span();

    // const Int32 item_rank = work_group.activeWorkItemRankInGroup();

    // Int32 tid = item_rank & (WSIZE - 1);
    // Int32 wid = item_rank / WSIZE;
    // Int32 wnum = group_size / WSIZE;
    // Int32 j, k;

    // for (j = item_rank; j < SH_ROW; j += group_size) {
    //   local_table[j] = -1;
    // }
    // if (item_rank == 0) {
    //   local_nnz[0] = 0;
    // }
    // work_group.barrier();
    // Int32 acol, bcol, hash, old;
    // Int32 rid = bins_view[bin_offset + group_rank];
    // for (j = arpt_view[rid] + wid; j < arpt_view[rid + 1]; j += wnum) {
    //   acol = acol_view[j];
    //   for (k = brpt_view[acol] + tid; k < brpt_view[acol + 1]; k += WSIZE) {
    //     bcol = bcol_view[k];
    //     hash = (bcol * HASH_SCALE) & (SH_ROW - 1);
    //     while (1) {
    //       old = atomicCAS(&local_table[hash], -1, bcol);
    //       if (old == -1) {
    //         ax::doAtomicAdd(&local_nnz[0], 1);
    //         break;
    //       } else if (old == bcol) {
    //         break;
    //       } else {
    //         hash = (hash + 1) & (SH_ROW - 1);
    //       }
    //     }
    //   }
    // }
    // work_group.barrier();

    // if (item_rank == 0) {
    //   row_nnz_view[rid] = local_nnz[0];
    // }

    const Int32 nb_items = work_group.nbActiveItem();
    const Int32 rid = bins_view[bin_offset + group_rank];

    for (Int32 item_rank = 0; item_rank < nb_items; ++item_rank) {
      Int32 j;

      for (j = item_rank; j < SH_ROW; j += group_size) {
        local_table[j] = -1;
      }
      if (item_rank == 0) {
        local_nnz[0] = 0;
      }
    }

    work_group.barrier();

    for (Int32 item_rank = 0; item_rank < nb_items; ++item_rank) {
      Int32 tid = item_rank & (WSIZE - 1);
      Int32 wid = item_rank / WSIZE;
      Int32 wnum = group_size / WSIZE;
      Int32 j, k;

      Int32 acol, bcol, hash, old;
      for (j = arpt_view[rid] + wid; j < arpt_view[rid + 1]; j += wnum) {
        acol = acol_view[j];
        for (k = brpt_view[acol] + tid; k < brpt_view[acol + 1]; k += WSIZE) {
          bcol = bcol_view[k];
          hash = (bcol * HASH_SCALE) & (SH_ROW - 1);
          while (1) {
            old = ax::doAtomicCAS(&local_table[hash], -1, bcol);
            if (old == -1) {
              local_nnz[0]++;
              break;
            } else if (old == bcol) {
              break;
            } else {
              hash = (hash + 1) & (SH_ROW - 1);
            }
          }
        }
      }
    }

    work_group.barrier();

    for (Int32 item_rank = 0; item_rank < nb_items; ++item_rank) {
      if (item_rank == 0) {
        row_nnz_view[rid] = local_nnz[0];
      }
    }
  };
}

template <int SH_ROW, int GROUP_SIZE> void ConnectivityMatMul::numericSharedHashTable(Int32 bin_index) {
  const ax::RunQueue &queue = m_meta->get_run_queue(bin_index);
  const Int32 bin_offset = m_meta->get_bin_offset(bin_index);
  const Int32 bin_size = m_meta->get_bin_size(bin_index);

  auto command = makeCommand(queue);
  auto arpt_view = ax::viewIn(command, *m_A.rpt);
  auto acol_view = ax::viewIn(command, *m_A.col);
  auto brpt_view = ax::viewIn(command, *m_B.rpt);
  auto bcol_view = ax::viewIn(command, *m_B.col);
  auto bins_view = ax::viewIn(command, *m_meta->bins);
  auto crpt_view = ax::viewInOut(command, *m_C.rpt);
  auto ccol_view = ax::viewInOut(command, *m_C.col);

  constexpr Int32 tsize = SH_ROW - 1;

  ax::LocalMemory<Int32, tsize> shared_col(command, tsize);
  ax::LocalMemory<Int32, 1> shared_offset(command, 1);

  ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, bin_size * GROUP_SIZE, bin_size, GROUP_SIZE);

  command << RUNCOMMAND_LAUNCH(ctx, loop_range, shared_col, shared_offset) {
    auto work_group = ctx.group();
    const Int32 group_rank = work_group.groupRank();
    const Int32 group_size = work_group.groupSize();

    auto local_col = shared_col.span();
    auto local_offset = shared_offset.span();

    // const Int32 item_rank = work_group.activeWorkItemRankInGroup();

    // Int32 tid = item_rank & (WSIZE - 1);
    // Int32 wid = item_rank / WSIZE;
    // Int32 wnum = group_size / WSIZE;
    // Int32 j, k;

    // for (j = item_rank; j < tsize; j += group_size) {
    //   local_col[j] = -1;
    // }
    // if (item_rank == 0) {
    //   local_offset[0] = 0;
    // }
    // work_group.barrier();

    // Int32 acol, bcol, hash, old;
    // Int32 rid = bins_view[group_rank];
    // Int32 c_offset = crpt_view[rid];
    // Int32 row_nnz = crpt_view[rid + 1] - crpt_view[rid];

    // for (j = arpt_view[rid] + wid; j < arpt_view[rid + 1]; j += wnum) {
    //   acol = acol_view[j];
    //   for (k = brpt_view[acol] + tid; k < brpt_view[acol + 1]; k += WSIZE) {
    //     bcol = bcol_view[k];
    //     hash = (bcol * HASH_SCALE) % tsize;
    //     while (1) {
    //       old = atomicCAS(&local_col[hash], -1, bcol);
    //       if (old == -1 || old == bcol) {
    //         break;
    //       } else {
    //         hash = (hash + 1) < tsize ? hash + 1 : 0;
    //       }
    //     }
    //   }
    // }

    // work_group.barrier();

    // // condense shared hash table
    // Int32 offset;
    // bool valid;
    // // #pragma unroll
    // for (j = 0; j < SH_ROW; j += GROUP_SIZE) {
    //   offset = j + item_rank;
    //   valid = offset < tsize;
    //   if (valid) {
    //     acol = local_col[offset];
    //     if (acol != -1) {
    //       offset = ax::doAtomicAdd(&local_offset[0], 1);
    //     }
    //   }
    //   work_group.barrier();
    //   if (valid && acol != -1) {
    //     local_col[offset] = acol;
    //   }
    // }

    // // count sort the result
    // work_group.barrier();
    // Int32 count, target;
    // for (j = item_rank; j < row_nnz; j += group_size) {
    //   target = local_col[j];
    //   count = 0;
    //   for (k = 0; k < row_nnz; ++k) {
    //     count += (unsigned int)(local_col[k] - target) >> 31;
    //   }
    //   ccol_view[c_offset + count] = local_col[j];
    // }

    const Int32 nb_items = work_group.nbActiveItem();

    for (Int32 item_idx = 0; item_idx < nb_items; ++item_idx) {
      const Int32 item_rank = work_group.isDevice() ? work_group.activeWorkItemRankInGroup() : item_idx;
      Int32 j;

      for (j = item_rank; j < tsize; j += group_size) {
        local_col[j] = -1;
      }
      if (item_rank == 0) {
        local_offset[0] = 0;
      }
    }

    work_group.barrier();

    for (Int32 item_idx = 0; item_idx < nb_items; ++item_idx) {
      const Int32 item_rank = work_group.isDevice() ? work_group.activeWorkItemRankInGroup() : item_idx;
      Int32 tid = item_rank & (WSIZE - 1);
      Int32 wid = item_rank / WSIZE;
      Int32 wnum = group_size / WSIZE;
      Int32 j, k;

      Int32 acol, bcol, hash, old;
      Int32 rid = bins_view[group_rank];
      Int32 c_offset = crpt_view[rid];
      Int32 row_nnz = crpt_view[rid + 1] - crpt_view[rid];

      for (j = arpt_view[rid] + wid; j < arpt_view[rid + 1]; j += wnum) {
        acol = acol_view[j];
        for (k = brpt_view[acol] + tid; k < brpt_view[acol + 1]; k += WSIZE) {
          bcol = bcol_view[k];
          hash = (bcol * HASH_SCALE) % tsize;
          while (1) {
            old = ax::doAtomicCAS(&local_col[hash], -1, bcol);
            if (old == -1 || old == bcol) {
              break;
            } else {
              hash = (hash + 1) < tsize ? hash + 1 : 0;
            }
          }
        }
      }
    }

    work_group.barrier();

    for (Int32 item_idx = 0; item_idx < nb_items; ++item_idx) {
      const Int32 item_rank = work_group.isDevice() ? work_group.activeWorkItemRankInGroup() : item_idx;
      Int32 j, acol;

      // condense shared hash table
      Int32 offset;
      bool valid;
      // #pragma unroll
      for (j = 0; j < SH_ROW; j += GROUP_SIZE) {
        offset = j + item_rank;
        valid = offset < tsize;
        if (valid) {
          acol = local_col[offset];
          if (acol != -1) {
            offset = ++local_offset[0];
          }
        }
        // if (work_group.isDevice()) {
        work_group.barrier();
        // }
        if (valid && acol != -1) {
          local_col[offset] = acol;
        }
      }
    }

    work_group.barrier();

    for (Int32 item_idx = 0; item_idx < nb_items; ++item_idx) {
      const Int32 item_rank = work_group.isDevice() ? work_group.activeWorkItemRankInGroup() : item_idx;
      Int32 j, k;

      Int32 rid = bins_view[group_rank];
      Int32 c_offset = crpt_view[rid];
      Int32 row_nnz = crpt_view[rid + 1] - crpt_view[rid];

      Int32 count, target;
      for (j = item_rank; j < row_nnz; j += group_size) {
        target = local_col[j];
        count = 0;
        for (k = 0; k < row_nnz; ++k) {
          count += (unsigned int)(local_col[k] - target) >> 31;
        }
        ccol_view[c_offset + count] = local_col[j];
      }
    }
  };
}

} // namespace Connectivix

#endif // CONNECTIVIX_CONNECTIVITY_MAT_MUL_H