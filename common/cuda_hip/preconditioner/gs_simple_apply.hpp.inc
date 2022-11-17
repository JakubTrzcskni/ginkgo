
namespace host_kernel {

template <bool prepermuted, int subwarp_size, int version, int b_s,
          typename ValueType, typename IndexType>
void apply_hbmc_v2(syn::value_list<int, b_s>,
                   std::shared_ptr<const Executor> exec,
                   const IndexType* l_diag_rows, const ValueType* l_diag_vals,
                   const preconditioner::parallel_block* p_block,
                   matrix::Dense<ValueType>* b_perm,
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
GKO_ENABLE_IMPLEMENTATION_SELECTION(select_apply_hbmc_v2, apply_hbmc_v2);

template <bool prepermuted, int version, int subwarp_size, typename ValueType,
          typename IndexType>
void apply_hbmc(syn::value_list<int, subwarp_size>,
                std::shared_ptr<const Executor> exec,
                const IndexType* l_diag_rows, const ValueType* l_diag_vals,
                const preconditioner::parallel_block* p_block,
                matrix::Dense<ValueType>* b_perm, matrix::Dense<ValueType>* x,
                const int* diag_LUT, const int* subblock_LUT,
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
    const auto b_s = p_block->base_block_size_;
    if (version == 1) {
        kernel::apply_l_p_block_kernel<prepermuted, subwarp_size>
            <<<grid_size, block_size>>>(
                &(l_diag_rows[id_offs]),
                as_device_type(&(l_diag_vals[id_offs])),
                p_block->end_row_global_, num_rows_p_block, b_s,
                p_block->degree_of_parallelism_, p_block->residual_, num_rhs,
                as_device_type(b_perm->get_values()),
                as_device_type(x->get_values()), permutation_idxs, diag_LUT,
                subblock_LUT);
    } else if (version == 2 || version == 3 || version == 4) {
        // std::cout << "kernel launch v" << version << ": " << std::endl;
        select_apply_hbmc_v2(
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
GKO_ENABLE_IMPLEMENTATION_SELECTION(select_apply_hbmc, apply_hbmc);

template <bool prepermuted, int kernel_version, typename ValueType,
          typename IndexType>
void hbmc_kernel_version(syn::value_list<int, kernel_version>,
                         std::shared_ptr<const Executor> exec,
                         const IndexType* l_diag_rows,
                         const ValueType* l_diag_vals,
                         const preconditioner::parallel_block* p_block,
                         matrix::Dense<ValueType>* b_perm,
                         matrix::Dense<ValueType>* x_perm, const int* diag_LUT,
                         const int* subblock_LUT,
                         const IndexType* permutation_idxs = nullptr)
{
    const auto w = p_block->lvl_2_block_size_;
    select_apply_hbmc(
        kernel::hbmc_kernels(),
        [&](int compiled_subwarp_size) { return compiled_subwarp_size == w; },
        syn::value_list<int, prepermuted, kernel_version>(), syn::type_list<>(),
        exec, l_diag_rows, l_diag_vals, p_block, b_perm, x_perm, diag_LUT,
        subblock_LUT, permutation_idxs);
}
GKO_ENABLE_IMPLEMENTATION_SELECTION(select_hbmc_kernel_version,
                                    hbmc_kernel_version);
}  // namespace host_kernel


template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const IndexType* l_diag_rows, const ValueType* l_diag_vals,
                  const IndexType* l_spmv_row_ptrs,
                  const IndexType* l_spmv_col_idxs,
                  const ValueType* l_spmv_vals,
                  const IndexType* permutation_idxs,
                  const preconditioner::storage_scheme& storage_scheme,
                  matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x,
                  int kernel_ver)
{
    GKO_ASSERT(!storage_scheme.symm_);
    const auto block_ptrs = storage_scheme.forward_solve_;
    const auto num_blocks = storage_scheme.num_blocks_;
    const auto num_rhs = b->get_size()[1];
    const auto num_rows = b->get_size()[0];

    auto diag_LUT = gko::array<gko::int32>();
    auto subblock_LUT = gko::array<gko::int32>();
    if (kernel_ver != 9) {
        diag_LUT = gko::array<gko::int32>(exec, max_b_s + 1);
        exec->copy_from<gko::int32>(exec->get_master().get(),
                                    static_cast<gko::size_type>(max_b_s + 1),
                                    diag_lut.data(), diag_LUT.get_data());
        subblock_LUT = gko::array<gko::int32>(exec, max_nz_block + 1);
        exec->copy_from<gko::int32>(
            exec->get_master().get(),
            static_cast<gko::size_type>(max_nz_block + 1), sub_block_lut.data(),
            subblock_LUT.get_data());
    }
    auto alpha = gko::initialize<gko::matrix::Dense<ValueType>>({-1.}, exec);
    auto beta = gko::initialize<gko::matrix::Dense<ValueType>>({1.}, exec);

    for (auto block = 0; block < num_blocks; block += 2) {
        auto p_block = static_cast<preconditioner::parallel_block*>(
            block_ptrs[block].get());

        host_kernel::select_hbmc_kernel_version(
            kernel::hbmc_kernel_version(),
            [&](int compiled_version) {
                return compiled_version == kernel_ver;
            },
            syn::value_list<int, false>(), syn::type_list<>(), exec,
            l_diag_rows, l_diag_vals, p_block, b, x, diag_LUT.get_const_data(),
            subblock_LUT.get_const_data(), permutation_idxs);

        if (block < num_blocks - 1) {
            auto spmv_block = static_cast<preconditioner::spmv_block*>(
                block_ptrs[block + 1].get());
            const auto spmv_size_row =
                spmv_block->end_row_global_ - spmv_block->start_row_global_;
            const auto spmv_size_col =
                spmv_block->end_col_global_ - spmv_block->start_col_global_;
            const auto spmv_nnz =
                l_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_ +
                                spmv_size_row];

            auto tmp_csr = gko::matrix::Csr<ValueType, IndexType>::create_const(
                exec, gko::dim<2>{spmv_size_row, spmv_size_col},
                gko::array<ValueType>::const_view(
                    exec, spmv_nnz,
                    &(l_spmv_vals[spmv_block->val_storage_id_])),
                gko::array<IndexType>::const_view(
                    exec, spmv_nnz,
                    &(l_spmv_col_idxs[spmv_block->val_storage_id_])),
                gko::array<IndexType>::const_view(
                    exec, spmv_size_row + 1,
                    &(l_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_])));
            auto tmp_b =
                b->create_submatrix(gko::span{spmv_block->start_row_global_,
                                              spmv_block->end_row_global_},
                                    gko::span{0, num_rhs});

            const auto perm_view = gko::array<IndexType>::view(
                exec, spmv_size_col,
                const_cast<IndexType*>(
                    &permutation_idxs[spmv_block->start_col_global_]));

            auto tmp_x = x->row_gather(&perm_view);

            csr::advanced_spmv(exec, lend(alpha), lend(tmp_csr), lend(tmp_x),
                               lend(beta), lend(tmp_b));
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL);