/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/preconditioner/gauss_seidel_kernels.hpp"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/allocator.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/merging.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/components/warp_blas.cuh"
#include "cuda/matrix/csr_kernels.cu"


namespace gko {
namespace kernels {
namespace cuda {
namespace gauss_seidel {
namespace {

// source: jacobi_kernels.cu
//  a total of 32 warps (1024 threads)
constexpr int default_num_warps = 32;
// with current architectures, at most 32 warps can be scheduled per SM (and
// current GPUs have at most 84 SMs)
constexpr int default_grid_size = 32 * 32 * 128;


}  // namespace
#include "common/cuda_hip/preconditioner/gauss_seidel_kernels.hpp.inc"


template <typename IndexType>
void get_degree_of_nodes(std::shared_ptr<const CudaExecutor> exec,
                         const IndexType num_vertices,
                         const IndexType* const row_ptrs,
                         IndexType* const degrees)
{
    const auto block_size = config::max_block_size;
    const auto grid_size = ceildiv(num_vertices, block_size);
    kernel::get_degree_of_nodes_kernel<<<grid_size, block_size>>>(
        num_vertices, row_ptrs, degrees);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_DEGREE_OF_NODES_KERNEL);

template <typename ValueType>
void ref_apply(std::shared_ptr<const CudaExecutor> exec, const LinOp* solver,
               const matrix::Dense<ValueType>* alpha,
               const matrix::Dense<ValueType>* b,
               const matrix::Dense<ValueType>* beta,
               matrix::Dense<ValueType>* x)
{
    solver->apply(alpha, b, beta, x);
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_APPLY_KERNEL);

template <typename ValueType>
void ref_simple_apply(std::shared_ptr<const CudaExecutor> exec,
                      const LinOp* solver, const matrix::Dense<ValueType>* b,
                      matrix::Dense<ValueType>* x)
{
    solver->apply(b, x);
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Csr<ValueType, IndexType>* A,
           const matrix::Dense<ValueType>* alpha,
           const matrix::Dense<ValueType>* rhs,
           const matrix::Dense<ValueType>* beta,
           matrix::Dense<ValueType>* x) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const CudaExecutor> exec,
                  const IndexType* l_diag_rows, const ValueType* l_diag_vals,
                  const IndexType* l_spmv_row_ptrs,
                  const IndexType* l_spmv_col_idxs,
                  const ValueType* l_spmv_vals,
                  const IndexType* permutation_idxs,
                  const preconditioner::storage_scheme& storage_scheme,
                  matrix::Dense<ValueType>* b_perm, matrix::Dense<ValueType>* x)
{
    std::cout << "apply cuda" << std::endl;
    GKO_ASSERT(!storage_scheme.symm_);
    const auto block_ptrs = storage_scheme.forward_solve_;
    const auto num_blocks = storage_scheme.num_blocks_;
    const auto num_rhs = b_perm->get_size()[1];
    const auto num_rows = b_perm->get_size()[0];

    auto diag_LUT = gko::array<gko::int32>(exec, max_block_size + 1);
    // exec, make_const_array_view(exec->get_master(), max_block_size + 1,
    //                             diag_lut.data()));
    exec->copy_from<gko::int32>(exec->get_master().get(),
                                static_cast<gko::size_type>(max_block_size + 1),
                                diag_lut.data(), diag_LUT.get_data());
    // cudaMemcpy(diag_LUT.get_data(), diag_lut.data(), max_block_size + 1,
    //            cudaMemcpyHostToDevice);
    auto subblock_LUT =
        gko::array<gko::int32>(exec, get_nz_block(max_block_size) + 1);
    // exec, make_const_array_view(exec->get_master(),
    //                             get_nz_block(max_block_size) + 1,
    //                             sub_block_lut.data()));
    exec->copy_from<gko::int32>(
        exec->get_master().get(),
        static_cast<gko::size_type>(get_nz_block(max_block_size) + 1),
        sub_block_lut.data(), subblock_LUT.get_data());
    // cudaMemcpy(subblock_LUT.get_data(), sub_block_lut.data(),
    //            subblock_LUT.get_num_elems(), cudaMemcpyHostToDevice);

    auto first_p_block =
        static_cast<preconditioner::parallel_block*>(block_ptrs[0].get());

    // for now only w == warp size is supported
    GKO_ASSERT(first_p_block->lvl_2_block_size_ == config::warp_size);

    const auto num_involved_warps =
        (config::min_warps_per_block > first_p_block->degree_of_parallelism_)
            ? config::min_warps_per_block
            : first_p_block->degree_of_parallelism_;
    const auto num_involved_threads = num_involved_warps * config::warp_size;
    const auto block_size = (num_involved_threads > config::max_block_size)
                                ? config::max_block_size
                                : num_involved_threads;
    const auto grid_size = ceildiv(num_involved_threads, block_size);

    kernel::apply_l_p_block_kernel<<<block_size, grid_size>>>(
        l_diag_rows, as_cuda_type(l_diag_vals), first_p_block->end_row_global_,
        first_p_block->base_block_size_, first_p_block->lvl_2_block_size_,
        first_p_block->degree_of_parallelism_, first_p_block->residual_,
        num_rhs, as_cuda_type(b_perm->get_values()),
        as_cuda_type(x->get_values()), permutation_idxs,
        diag_LUT.get_const_data(), subblock_LUT.get_const_data());

    for (auto block = 1; block < num_blocks - 1; block += 2) {
        auto spmv_block =
            static_cast<preconditioner::spmv_block*>(block_ptrs[block].get());
        const auto spmv_size_row =
            spmv_block->end_row_global_ - spmv_block->start_row_global_;
        const auto spmv_size_col =
            spmv_block->end_col_global_ - spmv_block->start_col_global_;
        const auto spmv_nnz =
            l_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_ + spmv_size_row];

        auto tmp_csr = gko::matrix::Csr<ValueType, IndexType>::create_const(
            exec, gko::dim<2>{spmv_size_row, spmv_size_col},
            gko::array<ValueType>::const_view(
                exec, spmv_nnz, &(l_spmv_vals[spmv_block->val_storage_id_])),
            gko::array<IndexType>::const_view(
                exec, spmv_nnz,
                &(l_spmv_col_idxs[spmv_block->val_storage_id_])),
            gko::array<IndexType>::const_view(
                exec, spmv_size_row + 1,
                &(l_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_])));
        auto tmp_b_perm =
            b_perm->create_submatrix(gko::span{spmv_block->start_row_global_,
                                               spmv_block->end_row_global_},
                                     gko::span{0, num_rhs});

        auto tmp_x = gko::matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{spmv_size_col, num_rhs});
        const auto block_size_spmv =
            (config::max_block_size < spmv_size_col)
                ? config::max_block_size
                : (config::min_warps_per_block * config::warp_size);
        const auto grid_size_spmv = ceildiv(spmv_size_col, block_size);

        kernel::prepare_x_kernel<<<block_size_spmv, grid_size_spmv>>>(
            x->get_const_values(), tmp_x->get_values(), permutation_idxs,
            num_rhs, spmv_block->start_col_global_, num_rows, spmv_size_col);

        auto alpha =
            gko::initialize<gko::matrix::Dense<ValueType>>({-1.}, exec);
        auto beta = gko::initialize<gko::matrix::Dense<ValueType>>({1.}, exec);

        csr::advanced_spmv(exec, lend(alpha), lend(tmp_csr), lend(tmp_x),
                           lend(beta), lend(tmp_b_perm));


        auto p_block = static_cast<preconditioner::parallel_block*>(
            block_ptrs[block + 1].get());
        auto id_offs = p_block->val_storage_id_;
        const auto num_involved_warps =
            (config::min_warps_per_block > p_block->degree_of_parallelism_)
                ? config::min_warps_per_block
                : p_block->degree_of_parallelism_;
        const auto num_involved_threads =
            num_involved_warps * config::warp_size;
        const auto block_size_p =
            (num_involved_threads > config::max_block_size)
                ? config::max_block_size
                : num_involved_threads;
        const auto grid_size_p = ceildiv(num_involved_threads, block_size);

        kernel::apply_l_p_block_kernel<<<block_size_p, grid_size_p>>>(
            &(l_diag_rows[id_offs]), as_cuda_type(&(l_diag_vals[id_offs])),
            p_block->end_row_global_, p_block->base_block_size_,
            p_block->lvl_2_block_size_, p_block->degree_of_parallelism_,
            p_block->residual_, num_rhs, as_cuda_type(b_perm->get_values()),
            as_cuda_type(x->get_values()), permutation_idxs,
            diag_LUT.get_const_data(), subblock_LUT.get_const_data());
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void get_coloring(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    array<IndexType>& vertex_colors, IndexType* max_color) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_COLORING_KERNEL);

template <typename ValueType, typename IndexType>
void get_block_coloring(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    const IndexType* block_ordering, const IndexType block_size,
    IndexType* vertex_colors, IndexType* max_color) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_BLOCK_COLORING_KERNEL);

template <typename ValueType, typename IndexType>
void assign_to_blocks(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    IndexType* block_ordering, const IndexType* degrees, int8* visited,
    const IndexType block_size) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_ASSIGN_TO_BLOCKS_KERNEL);

template <typename IndexType>
void get_secondary_ordering(std::shared_ptr<const CudaExecutor> exec,
                            IndexType* permutation_idxs,
                            preconditioner::storage_scheme& storage_scheme,
                            const IndexType base_block_size,
                            const IndexType lvl_2_block_size,
                            const IndexType* color_block_ptrs,
                            const IndexType max_color) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_SECONDARY_ORDERING_KERNEL);

template <typename ValueType, typename IndexType>
void setup_blocks(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* system_matrix,
                  const IndexType* permutation_idxs,
                  const IndexType* inv_permutation_idxs,
                  preconditioner::storage_scheme& storage_scheme,
                  IndexType* l_diag_rows, IndexType* l_diag_mtx_col_idxs,
                  ValueType* l_diag_vals, IndexType* l_spmv_row_ptrs,
                  IndexType* l_spmv_col_idxs, IndexType* l_spmv_mtx_col_idxs,
                  ValueType* l_spmv_vals, IndexType* u_diag_rows,
                  IndexType* u_diag_mtx_col_idxs, ValueType* u_diag_vals,
                  IndexType* u_spmv_row_ptrs, IndexType* u_spmv_col_idxs,
                  IndexType* u_spmv_mtx_col_idxs,
                  ValueType* u_spmv_vals) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_SETUP_BLOCKS_KERNEL);

template <typename ValueType, typename IndexType>
void fill_with_vals(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    const IndexType* permutation_idxs,
    const preconditioner::storage_scheme& storage_scheme,
    const IndexType diag_num_elems, const IndexType* l_diag_rows,
    const IndexType* l_diag_mtx_col_idxs, ValueType* l_diag_vals,
    const IndexType* l_spmv_row_ptrs, const IndexType* l_spmv_col_idxs,
    const IndexType* l_spmv_mtx_col_idxs, ValueType* l_spmv_vals,
    const IndexType* u_diag_rows, const IndexType* u_diag_mtx_col_idxs,
    ValueType* u_diag_vals, const IndexType* u_spmv_row_ptrs,
    const IndexType* u_spmv_col_idxs, const IndexType* u_spmv_mtx_col_idxs,
    ValueType* u_spmv_vals) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_FILL_WITH_VALS_KERNEL);

}  // namespace gauss_seidel
}  // namespace cuda
}  // namespace kernels
}  // namespace gko