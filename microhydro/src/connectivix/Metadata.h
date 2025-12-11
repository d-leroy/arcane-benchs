#ifndef CONNECTIVIX_METADATA_H
#define CONNECTIVIX_METADATA_H

#include "CSR.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunQueue.h"

using namespace Arcane;
namespace ax = Arcane::Accelerator;

namespace Connectivix {

class Metadata {

public:
  Int32 M;
  Int32 N;

  bool is_acc;

  NumArray<Int32, MDDim1> *bins;         // size M
  NumArray<Int32, MDDim1> *bin_size;     // size NUM_BIN
  NumArray<Int32, MDDim1> *bin_offset;   // size NUM_BIN
  NumArray<Int32, MDDim1> *max_row_nnz;  // size 1
  NumArray<Int32, MDDim1> *total_nnz;    // size 1
  NumArray<Int32, MDDim1> *scan_storage; // size variable

  Ref<ax::RunQueue> *run_queues;

public:
  Metadata(CSR &C, ax::Runner &runner);
  Metadata(const Metadata &) = delete;
  Metadata &operator=(const Metadata &) = delete;
  ~Metadata();

  void allocate_rpt(CSR &C);
  void allocate(ax::Runner &m_runner);
  void release();

  void barrier() const;
  void barrier(const Int32 queue_index) const;

  ax::RunQueue &get_run_queue(Int32 queue_index);
  Int32 get_bin_offset(Int32 bin_index);
  Int32 get_bin_size(Int32 bin_index);

  std::string printBins() const;
};

} // namespace Connectivix

#endif