// template <bool prepermuted, int subwarp_size, typename ValueType,
//           typename IndexType, typename Closure>
// __device__ void adv_apply_ltr(
//     const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
//     const size_type end_row, const size_type num_rows_in_block,
//     const size_type b_s, const size_type w_storage, const size_type num_rhs,
//     const ValueType* __restrict__ b, ValueType* __restrict__ x,
//     const IndexType* __restrict__ perm_idxs, Closure scale)
// {
//     const auto tid = thread::get_thread_id_flat<int>();
//     const auto col_id = blockIdx.y;
//     const auto curr_b_s = (num_rows_in_block - tid * b_s) < b_s
//                               ? (num_rows_in_block - tid * b_s)
//                               : b_s;
//     auto row = 0;
//     auto idx_write = 0;
//     while (row < curr_b_s) {
//         ValueType tmp{0};
//         const auto write_offs = idx_write * w_storage;
//         const auto row_global = rows[write_offs];
//         if (row_global >= 0) {
//             if (row_global >= end_row) return;
//             auto i = 0;
//             auto idx_read = 0;
//             while (i < row) {
//                 const auto val_read_offs = write_offs + (i - row) *
//                 w_storage; const auto x_read_offs = idx_read * w_storage;
//                 const auto row_read = rows[x_read_offs];
//                 if (row_read >= 0) {
//                     const auto x_row_read =
//                         prepermuted ? row_read : perm_idxs[row_read];
//                     tmp -=
//                         vals[val_read_offs] * x[x_row_read * num_rhs +
//                         col_id];
//                 }
//                 ++i;
//                 idx_read += (i + 1);
//             }
//             const auto val = vals[write_offs];
//             const auto x_row_global =
//                 prepermuted ? row_global : perm_idxs[row_global];
//             tmp += prepermuted ? b[row_global * num_rhs + col_id] +
//                                      x[x_row_global * num_rhs + col_id]
//                                : b[row_global * num_rhs + col_id];
//             x[x_row_global * num_rhs + col_id] =
//                 scale(tmp, x[x_row_global * num_rhs + col_id]);
//         }
//         ++row;
//         idx_write += (row + 1);
//     }
// }

template <bool forward, bool prepermuted, int subwarp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ void adv_apply_tr(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type b_s, const size_type w_storage, const size_type num_rhs,
    const ValueType* __restrict__ b, ValueType* __restrict__ x,
    const IndexType* __restrict__ perm_idxs, Closure scale)
{
    const auto tid = thread::get_thread_id_flat<int>();
    const auto col_id = blockIdx.y;
    const auto curr_b_s = (num_rows_in_block - tid * b_s) < b_s
                              ? (num_rows_in_block - tid * b_s)
                              : b_s;
    auto row = 0;
    auto idx_write = 0;
    while (row < curr_b_s) {
        ValueType tmp{0};
        const auto write_offs = idx_write * w_storage;
        const auto row_global = rows[write_offs];
        if (row_global >= 0) {
            if (row_global >= end_row) return;
            auto i = 0;
            auto idx_read = 0;
            while (i < row) {
                const auto val_read_offs =
                    forward ? write_offs + (i - row) * w_storage
                            : write_offs - (i + 1) * w_storage;
                const auto x_read_offs = idx_read * w_storage;
                const auto row_read = rows[x_read_offs];
                if (row_read >= 0) {
                    const auto x_row_read =
                        prepermuted ? row_read : perm_idxs[row_read];
                    tmp -=
                        vals[val_read_offs] * x[x_row_read * num_rhs + col_id];
                }
                ++i;
                idx_read += (i + 1);
            }
            auto val = vals[write_offs];
            assert(val != zero<ValueType>());
            val = forward ? val : one<ValueType>() / val;
            const auto x_row_global =
                prepermuted ? row_global : perm_idxs[row_global];
            const auto x_id = x_row_global * num_rhs + col_id;
            const auto b_id = forward ? row_global * num_rhs + col_id : x_id;
            tmp += (prepermuted && forward) ? b[b_id] + x[x_id] : b[b_id];
            x[x_id] = scale(val * tmp, x[x_id]);
        }
        ++row;
        idx_write += (row + 1);
    }
}


template <bool forward, bool prepermuted, int subwarp_size, typename ValueType,
          typename IndexType>
__global__ void adv_apply_kernel(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type b_s, const size_type nz_p_b, const size_type num_p_blocks,
    const bool res_blocks, const size_type num_rhs,
    const ValueType* __restrict__ b_perm, ValueType* __restrict__ x,
    const IndexType* __restrict__ perm_idxs)
{
    const auto tid = thread::get_thread_id_flat<int>();
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto local_tid = subwarp.thread_rank();
    const auto subwarp_id = thread::get_subwarp_id_flat<subwarp_size, int>();

    if (subwarp_id >= num_p_blocks || tid * b_s >= num_rows_in_block) return;
    const auto stride = nz_p_b * subwarp_size;
    const auto base_offs = subwarp_id * stride;
    if (subwarp_id == (num_p_blocks - 1) && res_blocks) {
        // apply base_block_agg
        const auto stride_base_block = nz_p_b;
        const auto base_block_offs = base_offs + local_tid * stride_base_block;
        adv_apply_tr<forward, prepermuted, subwarp_size>(
            &(rows[base_block_offs]), &(vals[base_block_offs]), end_row,
            num_rows_in_block, b_s, one<size_type>(), num_rhs, b_perm, x,
            perm_idxs,
            [](const ValueType& x, const ValueType& y) { return x; });
    } else {
        // apply lvl 1
        adv_apply_tr<forward, prepermuted, subwarp_size>(
            &(rows[base_offs + local_tid]), &(vals[base_offs + local_tid]),
            end_row, num_rows_in_block, b_s, subwarp_size, num_rhs, b_perm, x,
            perm_idxs,
            [](const ValueType& x, const ValueType& y) { return x; });
    }
}