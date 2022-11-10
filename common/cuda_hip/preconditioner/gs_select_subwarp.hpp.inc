template <bool prepermuted, int version, int subwarp_size, typename ValueType,
          typename IndexType>
void const_apply_hbmc(syn::value_list<int, subwarp_size>,
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
    GKO_ASSERT(version >= 2);
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
    const auto b_s = p_block->base_block_size_;
    if (version == 2 || version == 3 || version == 4) {
        select_const_apply_hbmc_v2(
            kernel::hbmc_block_sizes(),
            [&](int compiled_b_s) { return compiled_b_s == b_s; },
            syn::value_list<int, prepermuted, subwarp_size, version>(),
            syn::type_list<>(), exec, l_diag_rows, l_diag_vals, p_block, b_perm,
            x, diag_LUT, subblock_LUT, permutation_idxs);
    } else if (version == 5) {
        kernel::apply_l_p_block_kernel_v5<prepermuted, subwarp_size>
            <<<grid_size, block_size>>>(
                &(l_diag_rows[id_offs]),
                as_device_type(&(l_diag_vals[id_offs])),
                p_block->end_row_global_, num_rows_p_block, b_s,
                p_block->degree_of_parallelism_, p_block->residual_, num_rhs,
                as_device_type(b_perm->get_const_values()),
                as_device_type(x->get_values()), permutation_idxs, diag_LUT,
                subblock_LUT);
    } else if (version == 6) {
        kernel::apply_l_p_block_kernel_v6<prepermuted, subwarp_size>
            <<<grid_size, block_size>>>(
                &(l_diag_rows[id_offs]),
                as_device_type(&(l_diag_vals[id_offs])),
                p_block->end_row_global_, num_rows_p_block, b_s,
                p_block->degree_of_parallelism_, p_block->residual_, num_rhs,
                as_device_type(b_perm->get_const_values()),
                as_device_type(x->get_values()), permutation_idxs, diag_LUT,
                subblock_LUT);
    } else if (version == 7) {
        const auto grid_size_v7 = dim3(grid_size, num_rhs, 1);
        kernel::apply_l_p_block_kernel_v7<prepermuted, subwarp_size>
            <<<grid_size_v7, block_size>>>(
                &(l_diag_rows[id_offs]),
                as_device_type(&(l_diag_vals[id_offs])),
                p_block->end_row_global_, num_rows_p_block, b_s,
                p_block->degree_of_parallelism_, p_block->residual_, num_rhs,
                as_device_type(b_perm->get_const_values()),
                as_device_type(x->get_values()), permutation_idxs, diag_LUT,
                subblock_LUT);
    } else if (version == 8) {
        const auto grid_size_v8 = dim3(grid_size, num_rhs, 1);
        kernel::apply_l_p_block_kernel_v8<prepermuted, subwarp_size>
            <<<grid_size_v8, block_size>>>(
                &(l_diag_rows[id_offs]),
                as_device_type(&(l_diag_vals[id_offs])),
                p_block->end_row_global_, num_rows_p_block, b_s,
                p_block->degree_of_parallelism_, p_block->residual_, num_rhs,
                as_device_type(b_perm->get_const_values()),
                as_device_type(x->get_values()), permutation_idxs, diag_LUT,
                subblock_LUT);
    } else if (version == 9) {
        const auto grid_size_v9 = dim3(grid_size, num_rhs, 1);
        kernel::apply_l_p_block_kernel_v9<prepermuted, subwarp_size>
            <<<grid_size_v9, block_size>>>(
                &(l_diag_rows[id_offs]),
                as_device_type(&(l_diag_vals[id_offs])),
                p_block->end_row_global_, num_rows_p_block, b_s,
                p_block->nz_p_b_block_, p_block->degree_of_parallelism_,
                p_block->residual_, num_rhs,
                as_device_type(b_perm->get_const_values()),
                as_device_type(x->get_values()), permutation_idxs);
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}
GKO_ENABLE_IMPLEMENTATION_SELECTION(select_const_apply_hbmc, const_apply_hbmc);

template <bool forward, bool prepermuted, int subwarp_size, typename ValueType,
          typename IndexType>
void const_adv_apply_hbmc(syn::value_list<int, subwarp_size>,
                          std::shared_ptr<const Executor> exec,
                          const IndexType* l_diag_rows,
                          const ValueType* l_diag_vals,
                          const preconditioner::parallel_block* p_block,
                          const matrix::Dense<ValueType>* b_perm,
                          matrix::Dense<ValueType>* x,
                          const IndexType* permutation_idxs = nullptr)
{
    GKO_ASSERT(permutation_idxs || prepermuted);
    GKO_ASSERT(version >= 2);
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
    const auto b_s = p_block->base_block_size_;

    const auto grid_size_adv = dim3(grid_size, num_rhs, 1);
    kernel::adv_apply_kernel<forward, prepermuted, subwarp_size>
        <<<grid_size_adv, block_size>>>(
            &(l_diag_rows[id_offs]), as_device_type(&(l_diag_vals[id_offs])),
            p_block->end_row_global_, num_rows_p_block, b_s,
            p_block->nz_p_b_block_, p_block->degree_of_parallelism_,
            p_block->residual_, num_rhs,
            as_device_type(b_perm->get_const_values()),
            as_device_type(x->get_values()), permutation_idxs);
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_const_adv_apply_hbmc,
                                    const_adv_apply_hbmc);
