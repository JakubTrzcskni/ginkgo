template <bool prepermuted, int subwarp_size, int b_s, typename ValueType,
          typename IndexType, typename Closure>
__device__ void apply_ltr_v2(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type w_storage, const size_type num_rhs,
    const ValueType* __restrict__ b, ValueType* __restrict__ x,
    const IndexType* __restrict__ perm_idxs, const int* diag_LUT,
    const int* subblock_LUT, Closure scale)
{
    const auto tid = thread::get_thread_id_flat<int>();
    const auto curr_b_s = (num_rows_in_block - tid * b_s) < b_s
                              ? (num_rows_in_block - tid * b_s)
                              : b_s;
    const auto nz_p_b = (diag_LUT[curr_b_s] + 1);
    ValueType tmp_rhs[b_s];
    const auto base_row = rows[0];
    const auto last_row = rows[(nz_p_b - 1) * w_storage];
    const auto curr_w = (last_row - base_row) / (curr_b_s - 1);
    for (size_type k = 0; k < num_rhs; ++k) {
        for (auto i = 0; i < b_s; ++i) {
            tmp_rhs[i] = 0;
        }
        for (auto i = 0; i < nz_p_b; ++i) {
            const auto curr_offs = i * w_storage;
            const auto row = rows[curr_offs];
            if (row >= 0) {
                if (row >= end_row) return;
                const auto x_row = prepermuted ? row : perm_idxs[row];
                const auto rhs_id = (row - base_row) / curr_w;
                if (i == diag_LUT[subblock_LUT[i] + 1]) {
                    // on the diagonal
                    const auto val = vals[curr_offs];
                    assert(val != zero<ValueType>());
                    const auto inv_diag_val = one<ValueType>() / val;
                    tmp_rhs[rhs_id] += prepermuted ? b[row * num_rhs + k] +
                                                         x[x_row * num_rhs + k]
                                                   : b[row * num_rhs + k];
                    x[x_row * num_rhs + k] = scale(
                        inv_diag_val * tmp_rhs[rhs_id], x[x_row * num_rhs + k]);
                } else {
                    // off the diagonal
                    const auto read_offs =
                        diag_LUT[subblock_LUT[i] + 1] * w_storage;
                    const auto row_read = rows[read_offs];
                    if (row_read >= 0) {
                        const auto x_row_read =
                            prepermuted ? row_read : perm_idxs[row_read];
                        tmp_rhs[rhs_id] -=
                            vals[curr_offs] * x[x_row_read * num_rhs + k];
                    }
                }
            }
        }
    }
}

template <bool prepermuted, int subwarp_size, int b_s, typename ValueType,
          typename IndexType>
__global__ void apply_l_p_block_kernel_v2(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type num_p_blocks, const bool res_blocks,
    const size_type num_rhs, const ValueType* __restrict__ b_perm,
    ValueType* __restrict__ x, const IndexType* __restrict__ perm_idxs,
    const int* diag_LUT, const int* subblock_LUT)
{
    const auto tid = thread::get_thread_id_flat<int>();
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto local_tid = subwarp.thread_rank();
    const auto subwarp_id = thread::get_subwarp_id_flat<subwarp_size, int>();

    if (subwarp_id >= num_p_blocks || tid * b_s >= num_rows_in_block) return;
    const auto nz_p_b = diag_LUT[b_s] + 1;
    const auto stride = nz_p_b * subwarp_size;
    const auto base_offs = subwarp_id * stride;
    if (subwarp_id == (num_p_blocks - 1) && res_blocks) {
        // apply base_block_agg
        const auto stride_base_block = nz_p_b;
        const auto base_block_offs = base_offs + local_tid * stride_base_block;
        apply_ltr_v2<prepermuted, subwarp_size, b_s>(
            &(rows[base_block_offs]), &(vals[base_block_offs]), end_row,
            num_rows_in_block, one<size_type>(), num_rhs, b_perm, x, perm_idxs,
            diag_LUT, subblock_LUT,
            [](const ValueType& x, const ValueType& y) { return x; });
    } else {
        // apply lvl 1
        apply_ltr_v2<prepermuted, subwarp_size, b_s>(
            &(rows[base_offs + local_tid]), &(vals[base_offs + local_tid]),
            end_row, num_rows_in_block, subwarp_size, num_rhs, b_perm, x,
            perm_idxs, diag_LUT, subblock_LUT,
            [](const ValueType& x, const ValueType& y) { return x; });
    }
}

template <bool prepermuted, int subwarp_size, int b_s, typename ValueType,
          typename IndexType>
__global__ void apply_l_p_block_kernel_v4(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type num_p_blocks, const bool res_blocks,
    const size_type num_rhs, const ValueType* __restrict__ b_perm,
    ValueType* __restrict__ x, const IndexType* __restrict__ perm_idxs,
    const int* diag_LUT, const int* subblock_LUT)
{
    const auto tid = thread::get_thread_id_flat<int>();
    const auto l_tid = thread::get_local_thread_id<subwarp_size>();
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto local_tid = subwarp.thread_rank();
    const auto subwarp_id = thread::get_subwarp_id_flat<subwarp_size, int>();

    __shared__ int shared_diag_LUT[max_b_s + 1];
    __shared__ int shared_subblock_LUT[max_nz_block + 1];
    if (l_tid < max_nz_block + 1) {
        shared_subblock_LUT[l_tid] = subblock_LUT[l_tid];
    } else if (l_tid < max_nz_block + max_b_s + 2) {
        shared_diag_LUT[l_tid - max_nz_block - 1] =
            diag_LUT[l_tid - max_nz_block - 1];
    }

    if (subwarp_id >= num_p_blocks || tid * b_s >= num_rows_in_block) return;
    const auto nz_p_b = shared_diag_LUT[b_s] + 1;
    const auto stride = nz_p_b * subwarp_size;
    const auto base_offs = subwarp_id * stride;
    if (subwarp_id == (num_p_blocks - 1) && res_blocks) {
        // apply base_block_agg
        const auto stride_base_block = nz_p_b;
        const auto base_block_offs = base_offs + local_tid * stride_base_block;
        apply_ltr_v2<prepermuted, subwarp_size, b_s>(
            &(rows[base_block_offs]), &(vals[base_block_offs]), end_row,
            num_rows_in_block, one<size_type>(), num_rhs, b_perm, x, perm_idxs,
            shared_diag_LUT, shared_subblock_LUT,
            [](const ValueType& x, const ValueType& y) { return x; });

    } else {
        // apply lvl 1
        apply_ltr_v2<prepermuted, subwarp_size, b_s>(
            &(rows[base_offs + local_tid]), &(vals[base_offs + local_tid]),
            end_row, num_rows_in_block, subwarp_size, num_rhs, b_perm, x,
            perm_idxs, shared_diag_LUT, shared_subblock_LUT,
            [](const ValueType& x, const ValueType& y) { return x; });
    }
}