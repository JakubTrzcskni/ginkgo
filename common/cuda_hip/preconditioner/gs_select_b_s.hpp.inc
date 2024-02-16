template <bool prepermuted, int subwarp_size, int version, int b_s,
          typename ValueType, typename IndexType>
void const_apply_hbmc_v2(syn::value_list<int, b_s>,
                         std::shared_ptr<const Executor> exec,
                         const IndexType* l_diag_rows,
                         const ValueType* l_diag_vals,
                         const preconditioner::parallel_block* p_block,
                         const matrix::Dense<ValueType>* b_perm,
                         matrix::Dense<ValueType>* x, const int* diag_LUT,
                         const int* subblock_LUT,
                         const IndexType* permutation_idxs = nullptr)
{
    GKO_ASSERT(permutation_idxs || prepermuted);
    const auto num_rows_p_block =
        p_block->end_row_global_ - p_block->start_row_global_;
    const auto num_rhs = b_perm->get_size()[1];
    const auto num_rows = b_perm->get_size()[0];
    auto id_offs = p_block->val_storage_id_;
    const auto num_involved_subwarps = p_block->degree_of_parallelism_;
    const auto min_num_threads =
        config::min_warps_per_block * config::warp_size;
    const auto num_involved_threads =
        num_involved_subwarps * subwarp_size < min_num_threads
            ? min_num_threads
            : num_involved_subwarps * subwarp_size;
    const auto block_size = (num_involved_threads > config::max_block_size)
                                ? config::max_block_size
                                : num_involved_threads;
    const auto grid_size = ceildiv(num_involved_threads, block_size);

    if (version == 2) {
        kernel::apply_l_p_block_kernel_v2<prepermuted, subwarp_size, b_s>
            <<<grid_size, block_size>>>(
                &(l_diag_rows[id_offs]),
                as_device_type(&(l_diag_vals[id_offs])),
                p_block->end_row_global_, num_rows_p_block,
                p_block->degree_of_parallelism_, p_block->residual_, num_rhs,
                as_device_type(b_perm->get_const_values()),
                as_device_type(x->get_values()), permutation_idxs, diag_LUT,
                subblock_LUT);
    } else if (version == 3) {
        const auto grid_size_v3 = dim3(grid_size, num_rhs, 1);
        kernel::apply_l_p_block_kernel_v3<prepermuted, subwarp_size, b_s>
            <<<grid_size_v3, block_size>>>(
                &(l_diag_rows[id_offs]),
                as_device_type(&(l_diag_vals[id_offs])),
                p_block->end_row_global_, num_rows_p_block,
                p_block->degree_of_parallelism_, p_block->residual_, num_rhs,
                as_device_type(b_perm->get_const_values()),
                as_device_type(x->get_values()), permutation_idxs, diag_LUT,
                subblock_LUT);
    } else if (version == 4) {
        kernel::apply_l_p_block_kernel_v4<prepermuted, subwarp_size, b_s>
            <<<grid_size, block_size>>>(
                &(l_diag_rows[id_offs]),
                as_device_type(&(l_diag_vals[id_offs])),
                p_block->end_row_global_, num_rows_p_block,
                p_block->degree_of_parallelism_, p_block->residual_, num_rhs,
                as_device_type(b_perm->get_const_values()),
                as_device_type(x->get_values()), permutation_idxs, diag_LUT,
                subblock_LUT);
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}
GKO_ENABLE_IMPLEMENTATION_SELECTION(select_const_apply_hbmc_v2,
                                    const_apply_hbmc_v2);