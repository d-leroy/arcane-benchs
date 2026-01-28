#include "Metadata.h"
#include "arcane/accelerator/Runner.h"
#include "arcane/accelerator/core/DeviceInfo.h"
#include "define.h"

namespace Connectivix {

Metadata::Metadata(CSR &C, ax::Runner &runner) {
  is_acc = ax::isAcceleratorPolicy(runner.executionPolicy());
  if (is_acc) {
    available_shmem = runner.deviceInfo().sharedMemoryPerBlock();
  } else {
    available_shmem = -1;
  }
  allocate_rpt(C);
  run_queues = new Ref<ax::RunQueue>[NUM_BIN];
  for (int i = 0; i < NUM_BIN; ++i) {
    run_queues[i] = ax::makeQueueRef(runner);
    run_queues[i].get()->setAsync(true);
  }
}

void Metadata::allocate_rpt(CSR &C) {
  M = C.M;
  N = C.N;
  if (is_acc) {
    C.rpt = new NumArray<Int32, MDDim1>(C.M + 1, eMemoryRessource::Device);
    total_nnz = new NumArray<Int32, MDDim1>(1, eMemoryRessource::HostPinned);
    max_row_nnz = new NumArray<Int32, MDDim1>(1, eMemoryRessource::HostPinned);
    scan_storage = C.rpt;
  } else {
    C.rpt = new NumArray<Int32, MDDim1>(C.M + 1, eMemoryRessource::Host);
    total_nnz = new NumArray<Int32, MDDim1>(1, eMemoryRessource::Host);
    max_row_nnz = new NumArray<Int32, MDDim1>(1, eMemoryRessource::Host);
    scan_storage = new NumArray<Int32, MDDim1>(C.M + 1, eMemoryRessource::Host);

    (*total_nnz)[0] = 0;
    (*max_row_nnz)[0] = 0;
  }
}

void Metadata::allocate() {
  if (is_acc) {
    bins = new NumArray<Int32, MDDim1>(M, eMemoryRessource::Device);
    bin_size = new NumArray<Int32, MDDim1>(NUM_BIN, eMemoryRessource::HostPinned);
    bin_offset = new NumArray<Int32, MDDim1>(NUM_BIN, eMemoryRessource::HostPinned);
    bins->fill(0, run_queues[0].get());
    bin_size->fillHost(0);
    bin_offset->fillHost(0);
  } else {
    bins = new NumArray<Int32, MDDim1>(M, eMemoryRessource::Host);
    bin_size = new NumArray<Int32, MDDim1>(NUM_BIN, eMemoryRessource::Host);
    bin_offset = new NumArray<Int32, MDDim1>(NUM_BIN, eMemoryRessource::Host);
    bins->fillHost(0);
    bin_size->fillHost(0);
    bin_offset->fillHost(0);
  }
}

void Metadata::release() {
  if (run_queues != nullptr) {
    for (int i = 0; i < NUM_BIN; ++i) {
      run_queues[i]._release();
    }
    delete[] run_queues;
    run_queues = nullptr;
  }

  delete bins;
  delete bin_size;
  delete total_nnz;
  delete bin_offset;
  if (is_acc) {
    scan_storage = nullptr;
  } else {
    delete scan_storage;
  }
}

ax::RunQueue &Metadata::get_run_queue(Int32 queue_index) {
  return *(run_queues[queue_index].get());
}

Int32 Metadata::get_bin_offset(Int32 bin_index) {
  return (*bin_offset)[bin_index];
}

Int32 Metadata::get_bin_size(Int32 bin_index) {
  return (*bin_size)[bin_index];
}

void Metadata::barrier() const {
  for (int i = 0; i < NUM_BIN; ++i) {
    (*run_queues[i].get()).barrier();
  }
}

void Metadata::barrier(const Int32 queue_index) const {
  (*run_queues[queue_index].get()).barrier();
}

std::string Metadata::print_bins(NumArray<Int32, MDDim1> &nnz) const {
  NumArray<Int32, MDDim1> binsCopy(M, eMemoryRessource::Host);
  NumArray<Int32, MDDim1> nnzCopy(M, eMemoryRessource::Host);
  binsCopy.copy(*bins);
  nnzCopy.copy(nnz);
  std::stringstream sstream;
  sstream << "Bins:" << std::endl;
  for (Int32 binIdx = 0; binIdx < NUM_BIN; ++binIdx) {
    sstream << "[" << binIdx << "] ";
    auto from = (*bin_offset)[binIdx];
    auto to = from + (*bin_size)[binIdx];
    for (Int32 i = from; i < to; ++i) {
      sstream << binsCopy[i] << ":" << nnzCopy[binsCopy[i]] << " ";
    }
    sstream << std::endl;
  }

  return sstream.str();
}

Metadata::~Metadata() {
  release();
}

} // namespace Connectivix