template <bool prepermuted, int kernel_version, typename ValueType,
          typename IndexType>
void const_hbmc_kernel_version(syn::value_list<int, kernel_version>,
                               std::shared_ptr<const Executor> exec,
                               const IndexType* l_diag_rows,
                               const ValueType* l_diag_vals,
                               const preconditioner::parallel_block* p_block,
                               const matrix::Dense<ValueType>* b_perm,
                               matrix::Dense<ValueType>* x_perm,
                               const int* diag_LUT, const int* subblock_LUT,
                               const IndexType* permutation_idxs = nullptr)
{
    const auto w = p_block->lvl_2_block_size_;
    select_const_apply_hbmc(
        kernel::hbmc_kernels(),
        [&](int compiled_subwarp_size) { return compiled_subwarp_size == w; },
        syn::value_list<int, prepermuted, kernel_version>(), syn::type_list<>(),
        exec, l_diag_rows, l_diag_vals, p_block, b_perm, x_perm, diag_LUT,
        subblock_LUT, permutation_idxs);
}
GKO_ENABLE_IMPLEMENTATION_SELECTION(select_const_hbmc_kernel_version,
                                    const_hbmc_kernel_version);