

// NumArray<Int32, MDDim1> shared_bin_size(NUM_BIN, eMemoryRessource::HostPinned);
// NumArray<Int32, MDDim1> shared_bin_offset(NUM_BIN, eMemoryRessource::HostPinned);

// {
//   auto command = makeCommand(queue);

//   auto row_flop_view = ax::viewIn(command, *m_C.rpt);
//   auto bin_offset_view = ax::viewInOut(command, *m_meta->bin_offset);
//   auto bin_size_view = ax::viewInOut(command, *m_meta->bin_size);
//   auto bins_view = ax::viewOut(command, *m_meta->bins);

//   auto local_bin_size = ax::viewOut(command, shared_bin_size);

//   ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, group_size * nb_groups, nb_groups, group_size);

//   command << RUNCOMMAND_LAUNCH(ctx, loop_range) {
//     auto work_group = ctx.group();

//     const Int32 item_rank = work_group.activeWorkItemRankInGroup();
//     auto work_item = work_group.activeItem(0);
//     Int32 i = work_item.linearIndex();

//     if (item_rank < NUM_BIN) {
//       local_bin_size[item_rank] = 0;
//     }
//   };

//   queue->barrier();
// }

// {
//   auto command = makeCommand(queue);

//   auto row_flop_view = ax::viewIn(command, *m_C.rpt);
//   auto bin_offset_view = ax::viewInOut(command, *m_meta->bin_offset);
//   auto bin_size_view = ax::viewInOut(command, *m_meta->bin_size);
//   auto bins_view = ax::viewOut(command, *m_meta->bins);

//   auto local_bin_size = ax::viewInOut(command, shared_bin_size);

//   ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, group_size * nb_groups, nb_groups, group_size);

//   command << RUNCOMMAND_LAUNCH(ctx, loop_range) {
//     auto work_group = ctx.group();
//     constexpr Int32 range[NUM_BIN] = {32, 512, 1024, 2048, 4096, 8192, 12287, INT_MAX};
//     Int32 row_nnz, j;

//     auto work_item = work_group.activeItem(0);
//     Int32 i = work_item.linearIndex();

//     if (i < M) {
//       row_nnz = row_flop_view[i];
//       // #pragma unroll
//       for (j = 0; j < NUM_BIN; ++j) {
//         if (row_nnz <= range[j]) {
//           // The row belongs to bin j so we increment its local size.
//           ax::doAtomicAdd(local_bin_size[j], 1);
//           break;
//         }
//       }
//     }
//   };

//   queue->barrier();
// }

// std::cout << "Bin offsets:" << std::endl;
// for (int i = 0; i < NUM_BIN; i++) {
//   std::cout << "[" << i << "]: " << (*m_meta->bin_offset)[i] << std::endl;
// }

// {
//   auto command = makeCommand(queue);

//   auto bin_offset_view = ax::viewInOut(command, *m_meta->bin_offset);
//   auto bin_size_view = ax::viewInOut(command, *m_meta->bin_size);

//   auto local_bin_size = ax::viewInOut(command, shared_bin_size);
//   auto local_bin_offset = ax::viewOut(command, shared_bin_offset);

//   ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, group_size * nb_groups, nb_groups, group_size);

//   command << RUNCOMMAND_LAUNCH(ctx, loop_range) {
//     auto work_group = ctx.group();

//     const Int32 item_rank = work_group.activeWorkItemRankInGroup();

//     if (item_rank < NUM_BIN) {
//       // We accumulate in each bin.
//       local_bin_offset[item_rank] = ax::doAtomicAdd(bin_size_view[item_rank], local_bin_size[item_rank]);
//       local_bin_offset[item_rank] += bin_offset_view[item_rank];
//       local_bin_size[item_rank] = 0;
//     }
//   };

//   queue->barrier();
// }

// std::cout << "Shared bin offsets:" << std::endl;
// for (int i = 0; i < NUM_BIN; i++) {
//   std::cout << "[" << i << "]: " << shared_bin_offset[i] << std::endl;
// }

// {
//   auto command = makeCommand(queue);

//   auto row_flop_view = ax::viewIn(command, *m_C.rpt);
//   auto bin_offset_view = ax::viewInOut(command, *m_meta->bin_offset);
//   auto bin_size_view = ax::viewInOut(command, *m_meta->bin_size);
//   auto bins_view = ax::viewOut(command, *m_meta->bins);

//   auto local_bin_size = ax::viewInOut(command, shared_bin_size);
//   auto local_bin_offset = ax::viewInOut(command, shared_bin_offset);

//   ax::WorkGroupLoopRange loop_range = ax::makeWorkGroupLoopRange(command, group_size * nb_groups, nb_groups, group_size);

//   command << RUNCOMMAND_LAUNCH(ctx, loop_range) {
//     auto work_group = ctx.group();
//     constexpr Int32 range[NUM_BIN] = {32, 512, 1024, 2048, 4096, 8192, 12287, INT_MAX};
//     Int32 row_nnz, j;

//     auto work_item = work_group.activeItem(0);
//     Int32 i = work_item.linearIndex();

//     Int32 index;
//     if (i < M) {
//       // #pragma unroll
//       for (j = 0; j < NUM_BIN; ++j) {
//         if (row_nnz <= range[j]) {
//           index = ax::doAtomicAdd(local_bin_size[j], 1);
//           bins_view[local_bin_offset[j] + index] = i;
//           return;
//         }
//       }
//     }
//   };

//   queue->barrier();
// }