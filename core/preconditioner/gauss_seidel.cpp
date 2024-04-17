// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/gauss_seidel.hpp>


#include <cstring>
#include <list>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/reorder/rcm.hpp>


#include "core/base/allocator.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/permutation_kernels.hpp"
#include "core/preconditioner/gauss_seidel_kernels.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"
#include "core/utils/matrix_utils.hpp"

namespace gko {
namespace preconditioner {
namespace gauss_seidel {
namespace {
GKO_REGISTER_OPERATION(simple_apply, gauss_seidel::simple_apply);
GKO_REGISTER_OPERATION(prepermuted_simple_apply,
                       gauss_seidel::prepermuted_simple_apply);
GKO_REGISTER_OPERATION(advanced_apply, gauss_seidel::advanced_apply);
GKO_REGISTER_OPERATION(advanced_prepermuted_apply,
                       gauss_seidel::advanced_prepermuted_apply);
GKO_REGISTER_OPERATION(get_coloring, gauss_seidel::get_coloring);
GKO_REGISTER_OPERATION(get_block_coloring, gauss_seidel::get_block_coloring);
GKO_REGISTER_OPERATION(assign_to_blocks, gauss_seidel::assign_to_blocks);
GKO_REGISTER_OPERATION(get_permutation_from_coloring,
                       gauss_seidel::get_permutation_from_coloring);
GKO_REGISTER_OPERATION(get_secondary_ordering,
                       gauss_seidel::get_secondary_ordering);
GKO_REGISTER_OPERATION(invert_diagonal, jacobi::invert_diagonal);
GKO_REGISTER_OPERATION(get_degree_of_nodes, gauss_seidel::get_degree_of_nodes);
GKO_REGISTER_OPERATION(invert_permutation, permutation::invert);
GKO_REGISTER_OPERATION(setup_blocks, gauss_seidel::setup_blocks);
}  // namespace
}  // namespace gauss_seidel

// TODO
template <typename ValueType, typename IndexType>
GaussSeidel<ValueType, IndexType>& GaussSeidel<ValueType, IndexType>::operator=(
    const GaussSeidel& other)
{
    if (&other != this) {
        EnableLinOp<GaussSeidel>::operator=(other);
        lower_triangular_matrix_ = other.lower_triangular_matrix_;
        relaxation_factor_ = other.relaxation_factor_;
        parameters_ = other.parameters_;
    }
    return *this;
}


// TODO
template <typename ValueType, typename IndexType>
GaussSeidel<ValueType, IndexType>& GaussSeidel<ValueType, IndexType>::operator=(
    GaussSeidel&& other)
{
    if (&other != this) {
        EnableLinOp<GaussSeidel>::operator=(std::move(other));
        lower_triangular_matrix_ = std::move(other.lower_triangular_matrix_);
        relaxation_factor_ = std::exchange(other.relaxation_factor_, 1.0);
        parameters_ = std::exchange(other.parameters_, parameters_type{});
    }
    return *this;
}


template <typename ValueType, typename IndexType>
GaussSeidel<ValueType, IndexType>::GaussSeidel(const GaussSeidel& other)
    : GaussSeidel{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename IndexType>
GaussSeidel<ValueType, IndexType>::GaussSeidel(GaussSeidel&& other)
    : GaussSeidel{other.get_executor()}
{
    *this = std::move(other);
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::update_system(ValueType* values)
{}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                   LinOp* x) const
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    using Diagonal = matrix::Diagonal<ValueType>;
    using dim_type = gko::dim<2>::dimension_type;
    using Dense = matrix::Dense<ValueType>;

    bool permuted = permutation_idxs_.get_size() > 0;
    precision_dispatch_real_complex<ValueType>(
        [this, &permuted](auto dense_b, auto dense_x) {
            if (use_HBMC_) {
                if (prepermuted_input_) {
                    if (symmetric_preconditioner_) {
                        this->get_executor()->run(
                            gauss_seidel::make_advanced_prepermuted_apply(
                                l_diag_rows_.get_const_data(),
                                l_diag_vals_.get_const_data(),
                                l_spmv_row_ptrs_.get_const_data(),
                                l_spmv_col_idxs_.get_const_data(),
                                l_spmv_vals_.get_const_data(),
                                u_diag_rows_.get_const_data(),
                                u_diag_vals_.get_const_data(),
                                u_spmv_row_ptrs_.get_const_data(),
                                u_spmv_col_idxs_.get_const_data(),
                                u_spmv_vals_.get_const_data(),
                                permutation_idxs_.get_const_data(),
                                hbmc_storage_scheme_, dense_b, dense_x,
                                kernel_version_));
                    } else {
                        this->get_executor()->run(
                            gauss_seidel::make_prepermuted_simple_apply(
                                l_diag_rows_.get_const_data(),
                                l_diag_vals_.get_const_data(),
                                l_spmv_row_ptrs_.get_const_data(),
                                l_spmv_col_idxs_.get_const_data(),
                                l_spmv_vals_.get_const_data(),
                                hbmc_storage_scheme_,
                                permutation_idxs_.get_const_data(), dense_b,
                                dense_x, kernel_version_));
                    }
                } else {
                    auto b_perm =
                        as<Dense>(dense_b->row_permute(&permutation_idxs_));
                    if (symmetric_preconditioner_) {
                        this->get_executor()->run(
                            gauss_seidel::make_advanced_apply(
                                l_diag_rows_.get_const_data(),
                                l_diag_vals_.get_const_data(),
                                l_spmv_row_ptrs_.get_const_data(),
                                l_spmv_col_idxs_.get_const_data(),
                                l_spmv_vals_.get_const_data(),
                                u_diag_rows_.get_const_data(),
                                u_diag_vals_.get_const_data(),
                                u_spmv_row_ptrs_.get_const_data(),
                                u_spmv_col_idxs_.get_const_data(),
                                u_spmv_vals_.get_const_data(),
                                permutation_idxs_.get_const_data(),
                                hbmc_storage_scheme_, relaxation_factor_,
                                b_perm.get(), dense_x, kernel_version_));
                    } else {
                        this->get_executor()->run(
                            gauss_seidel::make_simple_apply(
                                l_diag_rows_.get_const_data(),
                                l_diag_vals_.get_const_data(),
                                l_spmv_row_ptrs_.get_const_data(),
                                l_spmv_col_idxs_.get_const_data(),
                                l_spmv_vals_.get_const_data(),
                                permutation_idxs_.get_const_data(),
                                hbmc_storage_scheme_, b_perm.get(), dense_x,
                                kernel_version_));
                    }
                }
            } else if (use_reference_ && !permuted) {
                if (symmetric_preconditioner_) {
                    GKO_NOT_IMPLEMENTED;
                } else {
                    lower_trs_->apply(dense_b, dense_x);
                }
            } else if (permuted) {
                const auto exec = host_exec_;
                auto b_perm =
                    share(as<Dense>(dense_b->row_permute(&permutation_idxs_)));
                auto x_perm =
                    share(as<Dense>(dense_x->row_permute(&permutation_idxs_)));
                const auto block_ptrs = color_ptrs_.get_const_data();
                const auto num_rows = lower_triangular_matrix_->get_size()[0];

                auto tmp_rhs_block = Dense::create(exec);

                for (auto color_block = 0;
                     color_block < color_ptrs_.get_size() - 1; color_block++) {
                    dim_type block_start = block_ptrs[color_block];
                    dim_type block_end = block_ptrs[color_block + 1];
                    dim_type block_size = block_end - block_start;

                    if (color_block == 0) {
                        const auto curr_b_block = Dense::create_const(
                            exec, dim<2>{block_size, b_perm->get_size()[1]},
                            gko::array<ValueType>::const_view(
                                exec, block_size * b_perm->get_size()[1],
                                &(b_perm->get_const_values()
                                      [block_start * b_perm->get_size()[1]])),
                            b_perm->get_size()[1]);
                        tmp_rhs_block->copy_from(curr_b_block);
                    }

                    auto curr_x_block = Dense::create(
                        exec, dim<2>{block_size, x_perm->get_size()[1]},
                        gko::make_array_view(
                            exec, block_size * x_perm->get_size()[1],
                            &(x_perm->get_values()[block_start *
                                                   x_perm->get_size()[1]])),
                        x_perm->get_size()[1]);

                    block_ptrs_[2 * color_block]->apply(tmp_rhs_block,
                                                        curr_x_block);

                    if (block_end < num_rows) {
                        dim_type next_block_start = block_ptrs[color_block + 1];
                        dim_type next_block_end = block_ptrs[color_block + 2];
                        dim_type next_block_size =
                            next_block_end - next_block_start;

                        const auto next_b_block = Dense::create_const(
                            exec,
                            dim<2>{next_block_size, b_perm->get_size()[1]},
                            gko::array<ValueType>::const_view(
                                exec, next_block_size * b_perm->get_size()[1],
                                &(b_perm->get_const_values()
                                      [next_block_start *
                                       b_perm->get_size()[1]])),
                            b_perm->get_size()[1]);

                        auto up_to_curr_x_block = Dense::create(
                            exec, dim<2>{block_end, x_perm->get_size()[1]},
                            gko::make_array_view(
                                exec, block_end * x_perm->get_size()[1],
                                &(x_perm->get_values()[0])),
                            x_perm->get_size()[1]);

                        tmp_rhs_block->copy_from(next_b_block);

                        auto one = gko::initialize<Dense>({1.0}, exec);
                        auto neg_one = gko::initialize<Dense>({-1.0}, exec);
                        block_ptrs_[2 * color_block + 1]->apply(
                            neg_one, up_to_curr_x_block, one, tmp_rhs_block);
                    }
                }
                dense_x->copy_from(std::move(as<Dense>(
                    x_perm->inverse_row_permute(&permutation_idxs_))));
            } else {
                GKO_NOT_SUPPORTED(this);
            }
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                   const LinOp* b,
                                                   const LinOp* beta,
                                                   LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            if (use_reference_) {
                lower_trs_->apply(dense_alpha, dense_b, dense_beta, dense_x);
            } else {
                auto x_clone = dense_x->clone();
                this->apply_impl(dense_b, x_clone.get());
                dense_x->scale(dense_beta);
                dense_x->add_scaled(dense_alpha, x_clone.get());
            }
        },
        alpha, b, beta, x);
}

template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
std::unique_ptr<matrix::SparsityCsr<ValueType, IndexType>>
GaussSeidel<ValueType, IndexType>::get_adjacency_matrix(
    matrix_data<ValueType, IndexType>& mat_data, bool is_symmetric)
{
    using SparsityMatrix = matrix::SparsityCsr<ValueType, IndexType>;
    using MatData = matrix_data<ValueType, IndexType>;

    auto exec = host_exec_;

    auto tmp = SparsityMatrix::create(exec);

    if (!is_symmetric) {
        utils::make_symmetric(mat_data);
    }
    tmp->read(mat_data);

    return give(tmp->to_adjacency_matrix());
}

template <typename ValueType, typename IndexType>
IndexType GaussSeidel<ValueType, IndexType>::get_coloring(
    matrix_data<ValueType, IndexType>& mat_data, bool is_symmetric)
{
    using SparsityMatrix = matrix::SparsityCsr<ValueType, IndexType>;
    using MatData = matrix_data<ValueType, IndexType>;

    auto exec = host_exec_;

    auto adjacency_matrix = SparsityMatrix::create(exec);
    adjacency_matrix = get_adjacency_matrix(mat_data, is_symmetric);

    vertex_colors_.fill(IndexType{-1});
    IndexType max_color{0};
    exec->run(gauss_seidel::make_get_coloring(adjacency_matrix.get(),
                                              vertex_colors_, &max_color));

    return max_color;
}

// TODO not finished
template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::initialize_blocks()
{
    using Diagonal = gko::matrix::Diagonal<ValueType>;
    using dim_type = gko::dim<2>::dimension_type;
    auto exec = host_exec_;
    const auto block_ptrs = color_ptrs_.get_const_data();
    const auto num_rows = lower_triangular_matrix_->get_size()[0];
    const auto num_of_colors = color_ptrs_.get_size() - 1;

    for (auto block_row = 0; block_row < num_of_colors; block_row++) {
        dim_type block_start = block_ptrs[block_row];
        dim_type block_end = block_ptrs[block_row + 1];
        dim_type block_size = block_end - block_start;


        auto curr_diag_block = lower_triangular_matrix_->create_submatrix(
            gko::span(block_start, block_end),
            gko::span(block_start, block_end));

        auto diag = gko::make_array_view(exec, block_size,
                                         curr_diag_block->get_values());

        auto curr_inv_diag_block = Diagonal::create(exec, block_size);
        auto inv_diag_view = gko::make_array_view(
            exec, block_size, curr_inv_diag_block->get_values());
        exec->run(gauss_seidel::make_invert_diagonal(diag, inv_diag_view));

        // TODO
        block_ptrs_.push_back(as<LinOp>(give(curr_inv_diag_block)));

        if (block_row < num_of_colors - 1) {
            dim_type next_block_start = block_end;
            dim_type next_block_end = block_ptrs[block_row + 2];
            auto curr_under_diag_block =
                lower_triangular_matrix_->create_submatrix(
                    gko::span(next_block_start, next_block_end),
                    gko::span(0, next_block_start));

            block_ptrs_.push_back(as<LinOp>(give(curr_under_diag_block)));
        }
    }
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix, bool skip_sorting)
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    const auto exec = host_exec_;
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    auto csr_matrix =
        convert_to_with_sorting<Csr>(exec, system_matrix, skip_sorting);
    const IndexType num_nodes = csr_matrix->get_size()[0];
    matrix_data<ValueType, IndexType> mat_data{csr_matrix->get_size()};


    if (!symmetric_preconditioner_) {
        csr_matrix->write(mat_data);
        if (use_reference_) {
            utils::make_lower_triangular(mat_data);
            lower_triangular_matrix_->read(mat_data);
            lower_trs_ =
                share(lower_trs_factory_->generate(lower_triangular_matrix_));
        } else {
            lower_trs_ = nullptr;
            auto max_color =  // colors start with 0 so there are max_colors
                              // + 1 different colors
                get_coloring(
                    mat_data);  // matrix data is made symmetric here for the
                                // purpose of creating an adjacency mtx / matrix
                                // is assumed to be non-symmetric
            color_ptrs_.resize_and_reset(max_color + 2);
            permutation_idxs_.resize_and_reset(num_nodes);
            exec->run(gauss_seidel::make_get_permutation_from_coloring(
                num_nodes, vertex_colors_.get_data(), max_color,
                color_ptrs_.get_data(), permutation_idxs_.get_data(),
                static_cast<IndexType*>(nullptr)));
            auto tmp = Csr::create(exec);
            tmp->read(mat_data);
            tmp = as<Csr>(tmp->permute(&permutation_idxs_));
            tmp->write(mat_data);
            utils::make_lower_triangular(mat_data);
            lower_triangular_matrix_->read(mat_data);
            initialize_blocks();
            GKO_ASSERT_EQ(block_ptrs_.size(), 2 * color_ptrs_.get_size() - 3);
        }

    } else {
        GKO_NOT_IMPLEMENTED;
        if (use_reference_) {
            // TODO
            auto diag = csr_matrix->extract_diagonal();
            auto diag_vals_view =
                make_array_view(exec, num_nodes, diag->get_values());
            exec->run(gauss_seidel::make_invert_diagonal(diag_vals_view,
                                                         diag_vals_view));
            // scale with omega
            auto omega =
                initialize<matrix::Dense<gko::remove_complex<ValueType>>>(
                    {relaxation_factor_}, exec);
            // csr_matrix->scale( (omega));

            // scale the lower triangular part with the diag inverse
            // diag->apply( (csr_matrix),  (csr_matrix));
            // make sure the diag of the upper part isn't scaled with omega
        }
    }
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::generate_HBMC(
    std::shared_ptr<const LinOp> system_matrix, bool skip_sorting)
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    auto exec = host_exec_;
    const auto is_gpu_executor = this->get_executor() != exec;

    auto csr_matrix =
        convert_to_with_sorting<Csr>(exec, system_matrix, skip_sorting);

    // std::cout << "reserve mem for mat_data:\n";
    matrix_data<ValueType, IndexType> mat_data{csr_matrix->get_size()};
    // std::cout << "mem reserved\n";
    csr_matrix->write(mat_data);
    auto adjacency_matrix =
        matrix::SparsityCsr<ValueType, IndexType>::create(exec);
    adjacency_matrix = get_adjacency_matrix(
        mat_data);  // this assumes that the matrix is not symmetric

    if (!storage_scheme_ready_) {
        auto block_ordering = generate_block_structure(
            adjacency_matrix.get(), base_block_size_,
            lvl2_block_size_);  // TODO a lot of functionality in this
                                // function, split up needed ?
    } else {
        auto num_nodes = csr_matrix->get_size()[0];
        auto max_color = (hbmc_storage_scheme_.num_blocks_ - 1) / 2;
        reserve_mem_for_block_structure(adjacency_matrix.get(),
                                        ceildiv(num_nodes, base_block_size_),
                                        base_block_size_, max_color + 1);

        if (preperm_mtx_) {
            permutation_idxs_.resize_and_reset(num_nodes);
            std::iota(permutation_idxs_.get_data(),
                      permutation_idxs_.get_data() + num_nodes, 0);
            inv_permutation_idxs_.resize_and_reset(num_nodes);
            exec->copy(num_nodes, permutation_idxs_.get_data(),
                       inv_permutation_idxs_.get_data());
        }
    }
    exec->run(gauss_seidel::make_setup_blocks(
        csr_matrix.get(), permutation_idxs_.get_const_data(),
        inv_permutation_idxs_.get_const_data(), hbmc_storage_scheme_,
        relaxation_factor_, l_diag_rows_.get_data(),
        l_diag_mtx_col_idxs_.get_data(), l_diag_vals_.get_data(),
        l_spmv_row_ptrs_.get_data(), l_spmv_col_idxs_.get_data(),
        l_spmv_mtx_col_idxs_.get_data(), l_spmv_vals_.get_data(),
        u_diag_rows_.get_data(), u_diag_mtx_col_idxs_.get_data(),
        u_diag_vals_.get_data(), u_spmv_row_ptrs_.get_data(),
        u_spmv_col_idxs_.get_data(), u_spmv_mtx_col_idxs_.get_data(),
        u_spmv_vals_.get_data(), preperm_mtx_));

    GKO_ASSERT(hbmc_storage_scheme_.num_blocks_ ==
               hbmc_storage_scheme_.forward_solve_.size());

    const auto d_exec = this->get_executor();
    l_diag_rows_.set_executor(d_exec);
    l_diag_mtx_col_idxs_.set_executor(d_exec);
    l_diag_vals_.set_executor(d_exec);
    l_spmv_row_ptrs_.set_executor(d_exec);
    l_spmv_col_idxs_.set_executor(d_exec);
    l_spmv_mtx_col_idxs_.set_executor(d_exec);
    l_spmv_vals_.set_executor(d_exec);
    if (symmetric_preconditioner_) {
        u_diag_rows_.set_executor(d_exec);
        u_diag_mtx_col_idxs_.set_executor(d_exec);
        u_diag_vals_.set_executor(d_exec);
        u_spmv_row_ptrs_.set_executor(d_exec);
        u_spmv_col_idxs_.set_executor(d_exec);
        u_spmv_mtx_col_idxs_.set_executor(d_exec);
        u_spmv_vals_.set_executor(d_exec);
    }
    permutation_idxs_.set_executor(d_exec);
    // inv_permutation_idxs_.set_executor(d_exec);
    // vertex_colors_.set_executor(d_exec);
    // color_ptrs_.set_executor(d_exec);

    // for testing only
    // lower_triangular_matrix_->copy_from(
    //     give(as<Csr>(csr_matrix->permute(&permutation_idxs_))));
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::reserve_mem_for_block_structure(
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    const IndexType num_base_blocks,
    const IndexType
        base_block_size,  // not needed, access the factory param instead?
    const IndexType num_colors)
{
    auto exec = host_exec_;
    const auto num_nodes = adjacency_matrix->get_size()[0];
    const auto nnz_triangle = adjacency_matrix->get_num_nonzeros() / 2 +
                              num_nodes;  // should work if matrix is symmetric

    // best case all blocks are dense
    auto diag_mem_requirement =
        num_base_blocks * kernels::precomputed_nz_p_b(base_block_size);
    if (use_padding_) {
        auto tmp = 0;
        const auto nz_per_lvl_1_block =
            lvl2_block_size_ * kernels::precomputed_nz_p_b(base_block_size);
        const auto lvl_1_size = lvl2_block_size_ * base_block_size_;
        if (storage_scheme_ready_) {
            for (auto i = 0; i < hbmc_storage_scheme_.num_blocks_; i += 2) {
                auto nodes_in_block = hbmc_storage_scheme_.forward_solve_[i]
                                          .get()
                                          ->end_row_global_ -
                                      hbmc_storage_scheme_.forward_solve_[i]
                                          .get()
                                          ->start_row_global_;
                tmp += nz_per_lvl_1_block * ceildiv(nodes_in_block, lvl_1_size);
            }
        } else {
            for (auto i = 0; i < num_colors; ++i) {
                auto nodes_in_color = color_ptrs_.get_const_data()[i + 1] -
                                      color_ptrs_.get_const_data()[i];
                tmp += nz_per_lvl_1_block * ceildiv(nodes_in_color, lvl_1_size);
            }
        }
        diag_mem_requirement = tmp;
    }

    // worst case all diag blocks are only a diagonal
    const auto l_spmv_val_col_mem_requirement =
        nnz_triangle - num_nodes;  // more memory than needed
    auto nodes_in_first_color =
        storage_scheme_ready_
            ? hbmc_storage_scheme_.forward_solve_[0].get()->end_row_global_
            : color_ptrs_.get_const_data()[1];
    const auto l_spmv_row_mem_requirement =
        num_nodes - nodes_in_first_color + num_colors - 1;  // optimal

    // std::cout << color_ptrs_.get_size() << "\n"
    //           << diag_mem_requirement << "\n"
    //           << l_spmv_row_mem_requirement << "\n"
    //           << l_spmv_val_col_mem_requirement << std::endl;
    l_diag_rows_.resize_and_reset(diag_mem_requirement);
    l_diag_rows_.fill(IndexType{-1});
    l_diag_mtx_col_idxs_.resize_and_reset(diag_mem_requirement);
    l_diag_mtx_col_idxs_.fill(IndexType{-1});
    l_diag_vals_.resize_and_reset(diag_mem_requirement);
    l_diag_vals_.fill(ValueType{0});
    l_spmv_row_ptrs_.resize_and_reset(l_spmv_row_mem_requirement);
    l_spmv_row_ptrs_.fill(IndexType{0});
    l_spmv_col_idxs_.resize_and_reset(l_spmv_val_col_mem_requirement);
    l_spmv_col_idxs_.fill(IndexType{0});
    l_spmv_mtx_col_idxs_.resize_and_reset(l_spmv_val_col_mem_requirement);
    l_spmv_mtx_col_idxs_.fill(IndexType{0});
    l_spmv_vals_.resize_and_reset(l_spmv_val_col_mem_requirement);
    l_spmv_vals_.fill(ValueType{0});

    const auto num_blocks = num_colors * 2 - 1;

    if (symmetric_preconditioner_) {
        const auto u_spmv_val_col_mem_requirement =
            l_spmv_val_col_mem_requirement;
        auto nodes_in_all_but_last_color =
            storage_scheme_ready_
                ? hbmc_storage_scheme_.backward_solve_[0]
                      .get()
                      ->start_row_global_
                : color_ptrs_.get_const_data()[color_ptrs_.get_size() - 2];
        const auto u_spmv_row_mem_requirement =
            +nodes_in_all_but_last_color + num_colors - 1;

        u_diag_rows_.resize_and_reset(diag_mem_requirement);
        u_diag_rows_.fill(IndexType{-1});
        u_diag_mtx_col_idxs_.resize_and_reset(diag_mem_requirement);
        u_diag_mtx_col_idxs_.fill(IndexType{-1});
        u_diag_vals_.resize_and_reset(diag_mem_requirement);
        u_diag_vals_.fill(ValueType{0});
        u_spmv_row_ptrs_.resize_and_reset(u_spmv_row_mem_requirement);
        u_spmv_row_ptrs_.fill(IndexType{0});
        u_spmv_col_idxs_.resize_and_reset(u_spmv_val_col_mem_requirement);
        u_spmv_col_idxs_.fill(IndexType{0});
        u_spmv_mtx_col_idxs_.resize_and_reset(u_spmv_val_col_mem_requirement);
        u_spmv_mtx_col_idxs_.fill(IndexType{0});
        u_spmv_vals_.resize_and_reset(u_spmv_val_col_mem_requirement);
        u_spmv_vals_.fill(ValueType{0});
    }

    if (!storage_scheme_ready_)
        hbmc_storage_scheme_ =
            storage_scheme(num_blocks, symmetric_preconditioner_);
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::recreate_block_ordering(
    const matrix::Csr<ValueType, IndexType>* system_matrix)
{
    if (use_padding_) {
    } else {
        if (system_matrix->get_size()[0] < base_block_size_ * lvl2_block_size_)
            return;
    }
}

template <typename ValueType, typename IndexType>
array<IndexType> GaussSeidel<ValueType, IndexType>::generate_block_structure(
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    const IndexType block_size, const IndexType lvl_2_block_size)
{
    auto exec = host_exec_;
    const IndexType num_nodes = adjacency_matrix->get_size()[0];
    const IndexType num_base_blocks = ceildiv(num_nodes, block_size);

    array<IndexType> block_ordering(exec, num_nodes);
    if (!preperm_mtx_) {
        array<IndexType> degrees(exec, num_nodes);
        array<int8> visited(exec, num_nodes);
        exec->run(gauss_seidel::make_get_degree_of_nodes(
            num_nodes, adjacency_matrix->get_const_row_ptrs(),
            degrees.get_data()));

        std::fill_n(visited.get_data(), num_nodes, int8{0});
        exec->run(gauss_seidel::make_assign_to_blocks(
            adjacency_matrix, block_ordering.get_data(),
            degrees.get_const_data(), visited.get_data(), block_size));
    } else {
        std::iota(block_ordering.get_data(),
                  block_ordering.get_data() + num_nodes, 0);
    }
    // TODO move to generate_HBMC / to gs kernels
    IndexType max_color = 0;
    vertex_colors_.fill(IndexType{-1});
    exec->run(gauss_seidel::make_get_block_coloring(
        adjacency_matrix, block_ordering.get_const_data(), block_size,
        vertex_colors_.get_data(), &max_color));

    color_ptrs_.resize_and_reset(max_color + 2);
    permutation_idxs_.resize_and_reset(num_nodes);
    inv_permutation_idxs_.resize_and_reset(num_nodes);

    exec->run(gauss_seidel::make_get_permutation_from_coloring(
        num_nodes, vertex_colors_.get_data(), max_color, color_ptrs_.get_data(),
        permutation_idxs_.get_data(), block_ordering.get_const_data()));


    reserve_mem_for_block_structure(adjacency_matrix, num_base_blocks,
                                    block_size, max_color + 1);

    if (preperm_mtx_) {
        permutation_idxs_.resize_and_reset(num_nodes);
        std::iota(permutation_idxs_.get_data(),
                  permutation_idxs_.get_data() + num_nodes, 0);
    }

    // secondary ordering
    exec->run(gauss_seidel::make_get_secondary_ordering(
        permutation_idxs_.get_data(), hbmc_storage_scheme_, block_size,
        lvl_2_block_size, color_ptrs_.get_const_data(), max_color, use_padding_,
        preperm_mtx_));

    if (preperm_mtx_) {
        permutation_idxs_.resize_and_reset(num_nodes);
        std::iota(permutation_idxs_.get_data(),
                  permutation_idxs_.get_data() + num_nodes, 0);
        exec->copy(num_nodes, permutation_idxs_.get_data(),
                   inv_permutation_idxs_.get_data());
    } else {
        exec->run(gauss_seidel::make_invert_permutation(
            permutation_idxs_.get_const_data(), num_nodes,
            inv_permutation_idxs_.get_data()));
    }


    return block_ordering;  // won't be needed at all
                            // return storage scheme maybe? or nothing
}

#define GKO_DECLARE_GAUSS_SEIDEL(ValueType, IndexType) \
    class GaussSeidel<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GAUSS_SEIDEL);

}  // namespace preconditioner
}  // namespace gko
