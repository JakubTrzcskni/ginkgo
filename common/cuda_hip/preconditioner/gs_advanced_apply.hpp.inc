
template <typename ValueType, typename IndexType>
void advanced_apply(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* l_diag_rows,
    const ValueType* l_diag_vals, const IndexType* l_spmv_row_ptrs,
    const IndexType* l_spmv_col_idxs, const ValueType* l_spmv_vals,
    const IndexType* u_diag_rows, const ValueType* u_diag_vals,
    const IndexType* u_spmv_row_ptrs, const IndexType* u_spmv_col_idxs,
    const ValueType* u_spmv_vals, const IndexType* permutation_idxs,
    const preconditioner::storage_scheme& storage_scheme,
    const remove_complex<ValueType> omega, matrix::Dense<ValueType>* b_perm,
    matrix::Dense<ValueType>* x, int kernel_version)
{
    GKO_ASSERT(storage_scheme.symm_);
    GKO_ASSERT(kernel_version == 9);
    const auto forward_solve = storage_scheme.forward_solve_;
    const auto backward_solve = storage_scheme.backward_solve_;

    const auto num_blocks = storage_scheme.num_blocks_;
    const auto num_rhs = b_perm->get_size()[1];
    const auto num_rows = b_perm->get_size()[0];
    auto alpha_spmv = initialize<matrix::Dense<ValueType>>({-1.}, exec);
    auto beta_spmv = initialize<matrix::Dense<ValueType>>({1.}, exec);
    auto beta_spmv_backward = initialize<matrix::Dense<ValueType>>({0.}, exec);

    // forward solve
    for (auto block = 0; block < num_blocks; block += 2) {
        auto p_block = static_cast<preconditioner::parallel_block*>(
            forward_solve[block].get());
        const auto w = p_block->lvl_2_block_size_;
        host_kernel::select_const_adv_apply_hbmc(
            kernel::hbmc_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size == w;
            },
            syn::value_list<int, true, false>(), syn::type_list<>(), exec,
            l_diag_rows, l_diag_vals, p_block, b_perm, x, permutation_idxs);

        if (block < num_blocks - 1) {
            auto spmv_block = static_cast<preconditioner::spmv_block*>(
                forward_solve[block + 1].get());
            const auto spmv_size_row =
                spmv_block->end_row_global_ - spmv_block->start_row_global_;
            const auto spmv_size_col =
                spmv_block->end_col_global_ - spmv_block->start_col_global_;
            const auto spmv_nnz =
                l_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_ +
                                spmv_size_row];

            auto tmp_csr = matrix::Csr<ValueType, IndexType>::create_const(
                exec, dim<2>{spmv_size_row, spmv_size_col},
                array<ValueType>::const_view(
                    exec, spmv_nnz,
                    &(l_spmv_vals[spmv_block->val_storage_id_])),
                array<IndexType>::const_view(
                    exec, spmv_nnz,
                    &(l_spmv_col_idxs[spmv_block->val_storage_id_])),
                array<IndexType>::const_view(
                    exec, spmv_size_row + 1,
                    &(l_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_])));

            auto tmp_b =
                b_perm->create_submatrix(span{spmv_block->start_row_global_,
                                              spmv_block->end_row_global_},
                                         span{0, num_rhs});

            const auto perm_view = array<IndexType>::view(
                exec, spmv_size_col,
                const_cast<IndexType*>(
                    &permutation_idxs[spmv_block->start_col_global_]));

            auto tmp_x = x->row_gather(&perm_view);

            csr::advanced_spmv(exec, alpha_spmv.get(), tmp_csr.get(),
                               tmp_x.get(), beta_spmv.get(), tmp_b.get());
        }
    }
    b_perm->fill(ValueType{0.});
    for (auto block = 0; block < num_blocks; block += 2) {
        auto p_block = static_cast<preconditioner::parallel_block*>(
            backward_solve[block].get());

        const auto w = p_block->lvl_2_block_size_;
        host_kernel::select_const_adv_apply_hbmc(
            kernel::hbmc_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size == w;
            },
            syn::value_list<int, false, false>(), syn::type_list<>(), exec,
            u_diag_rows, u_diag_vals, p_block, b_perm, x, permutation_idxs);

        if (block < num_blocks - 1) {
            auto spmv_block = static_cast<preconditioner::spmv_block*>(
                backward_solve[block + 1].get());
            const auto spmv_size_row =
                spmv_block->end_row_global_ - spmv_block->start_row_global_;
            const auto spmv_size_col =
                spmv_block->end_col_global_ - spmv_block->start_col_global_;
            const auto spmv_nnz =
                u_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_ +
                                spmv_size_row];

            auto tmp_csr = matrix::Csr<ValueType, IndexType>::create_const(
                exec, dim<2>{spmv_size_row, spmv_size_col},
                array<ValueType>::const_view(
                    exec, spmv_nnz,
                    &(u_spmv_vals[spmv_block->val_storage_id_])),
                array<IndexType>::const_view(
                    exec, spmv_nnz,
                    &(u_spmv_col_idxs[spmv_block->val_storage_id_])),
                array<IndexType>::const_view(
                    exec, spmv_size_row + 1,
                    &(u_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_])));
            GKO_ASSERT(spmv_block->start_col_global_ + spmv_size_col <
                       num_rows);
            const auto perm_view_in = array<IndexType>::view(
                exec, spmv_size_col,
                const_cast<IndexType*>(
                    &permutation_idxs[spmv_block->start_col_global_]));
            auto tmp_x_in = x->row_gather(&perm_view_in);

            auto tmp_b =
                b_perm->create_submatrix(span{spmv_block->start_row_global_,
                                              spmv_block->end_row_global_},
                                         span{0, num_rhs});

            /* std::cout << "block " << block << " row "
                      << spmv_block->start_row_global_ << " "
                      << spmv_block->end_row_global_ << " col "
                      << spmv_block->start_col_global_ << " "
                      << spmv_block->end_col_global_ << std::endl;
            std::cout << "tmp_csr" << std::endl;
            write(std::cout,  (tmp_csr), layout_type::coordinate);
            std::cout << "tmp_x_in" << std::endl;
            write(std::cout,  (tmp_x_in), layout_type::coordinate); */

            csr::advanced_spmv(exec, alpha_spmv.get(), tmp_csr.get(),
                               tmp_x_in.get(), beta_spmv_backward.get(),
                               tmp_b.get());
            /*  std::cout << "tmp_b" << std::endl;
             write(std::cout,  (tmp_b), layout_type::coordinate); */
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_ADVANCED_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void advanced_prepermuted_apply(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* l_diag_rows,
    const ValueType* l_diag_vals, const IndexType* l_spmv_row_ptrs,
    const IndexType* l_spmv_col_idxs, const ValueType* l_spmv_vals,
    const IndexType* u_diag_rows, const ValueType* u_diag_vals,
    const IndexType* u_spmv_row_ptrs, const IndexType* u_spmv_col_idxs,
    const ValueType* u_spmv_vals, const IndexType* permutation_idxs,
    const preconditioner::storage_scheme& storage_scheme,
    const matrix::Dense<ValueType>* b_perm, matrix::Dense<ValueType>* x_perm,
    int kernel_version)
{
    GKO_ASSERT(storage_scheme.symm_);
    GKO_ASSERT(kernel_version == 9);
    const auto forward_solve = storage_scheme.forward_solve_;
    const auto backward_solve = storage_scheme.backward_solve_;

    const auto num_blocks = storage_scheme.num_blocks_;
    const auto num_rhs = b_perm->get_size()[1];
    const auto num_rows = b_perm->get_size()[0];
    auto alpha_spmv =
        gko::initialize<gko::matrix::Dense<ValueType>>({-1.}, exec);
    auto beta_spmv_forward =
        gko::initialize<gko::matrix::Dense<ValueType>>({0.}, exec);
    auto beta_spmv_backward =
        gko::initialize<gko::matrix::Dense<ValueType>>({1.}, exec);
    // forward solve
    for (auto block = 0; block < num_blocks; block += 2) {
        auto p_block = static_cast<preconditioner::parallel_block*>(
            forward_solve[block].get());
        const auto w = p_block->lvl_2_block_size_;
        host_kernel::select_const_adv_apply_hbmc(
            kernel::hbmc_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size == w;
            },
            syn::value_list<int, true, true>(), syn::type_list<>(), exec,
            l_diag_rows, l_diag_vals, p_block, b_perm, x_perm,
            permutation_idxs);

        if (block < num_blocks - 1) {
            auto spmv_block = static_cast<preconditioner::spmv_block*>(
                forward_solve[block + 1].get());
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
            auto x_in = x_perm->create_submatrix(
                gko::span{spmv_block->start_col_global_,
                          spmv_block->end_col_global_},
                gko::span{0, num_rhs});
            auto x_out_tmp = x_perm->create_submatrix(
                gko::span{spmv_block->start_row_global_,
                          spmv_block->end_row_global_},
                gko::span{0, num_rhs});

            csr::advanced_spmv(exec, alpha_spmv.get(), tmp_csr.get(),
                               x_in.get(), beta_spmv_forward.get(),
                               x_out_tmp.get());
        }
    }

    for (auto block = 0; block < num_blocks; block += 2) {
        auto p_block = static_cast<preconditioner::parallel_block*>(
            backward_solve[block].get());

        const auto w = p_block->lvl_2_block_size_;
        host_kernel::select_const_adv_apply_hbmc(
            kernel::hbmc_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size == w;
            },
            syn::value_list<int, false, true>(), syn::type_list<>(), exec,
            u_diag_rows, u_diag_vals, p_block, x_perm, x_perm,
            permutation_idxs);

        if (block < num_blocks - 1) {
            auto spmv_block = static_cast<preconditioner::spmv_block*>(
                backward_solve[block + 1].get());
            const auto spmv_size_row =
                spmv_block->end_row_global_ - spmv_block->start_row_global_;
            const auto spmv_size_col =
                spmv_block->end_col_global_ - spmv_block->start_col_global_;
            const auto spmv_nnz =
                u_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_ +
                                spmv_size_row];

            auto tmp_csr = gko::matrix::Csr<ValueType, IndexType>::create_const(
                exec, gko::dim<2>{spmv_size_row, spmv_size_col},
                gko::array<ValueType>::const_view(
                    exec, spmv_nnz,
                    &(u_spmv_vals[spmv_block->val_storage_id_])),
                gko::array<IndexType>::const_view(
                    exec, spmv_nnz,
                    &(u_spmv_col_idxs[spmv_block->val_storage_id_])),
                gko::array<IndexType>::const_view(
                    exec, spmv_size_row + 1,
                    &(u_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_])));
            auto x_in = x_perm->create_submatrix(
                gko::span{spmv_block->start_col_global_,
                          spmv_block->end_col_global_},
                gko::span{0, num_rhs});
            auto x_out_tmp = x_perm->create_submatrix(
                gko::span{spmv_block->start_row_global_,
                          spmv_block->end_row_global_},
                gko::span{0, num_rhs});

            csr::advanced_spmv(exec, alpha_spmv.get(), tmp_csr.get(),
                               x_in.get(), beta_spmv_backward.get(),
                               x_out_tmp.get());
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_ADVANCED_PREPERMUTED_APPLY_KERNEL);
