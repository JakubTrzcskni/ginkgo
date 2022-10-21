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

#ifndef GKO_CORE_PRECONDITIONER_GAUSS_SEIDEL_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_GAUSS_SEIDEL_KERNELS_HPP_


#include <ginkgo/core/preconditioner/gauss_seidel.hpp>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/kernel_declaration.hpp"

namespace gko {
namespace kernels {

template <typename IndexType>
constexpr IndexType get_nz_block(const IndexType block_size)
{
    return (block_size * block_size - block_size) / 2 + block_size;
}

// Source: https://joelfilho.com/blog/2020/compile_time_lookup_tables_in_cpp/
template <gko::size_type Length, typename Generator, std::size_t... Indexes>
constexpr auto lut_impl(Generator&& f, std::index_sequence<Indexes...>)
{
    using content_type = decltype(f(std::size_t{0}));
    return std::array<content_type, Length>{{f(Indexes)...}};
}

template <gko::size_type Length, typename Generator>
constexpr auto lut(Generator&& f)
{
    return lut_impl<Length>(std::forward<Generator>(f),
                            std::make_index_sequence<Length>{});
}
constexpr auto max_b_s = 16;
constexpr auto max_nz_block = get_nz_block(max_b_s);
constexpr gko::int32 diag(gko::int32 n)
{
    gko::int32 result = 0;
    for (gko::int32 i = 0; i <= n; i++) {
        result += i;
    }
    return result - 1;
}
constexpr auto diag_lut = lut<max_b_s + 1>(diag);

constexpr gko::int32 sub_block(gko::int32 n)
{
    gko::int32 result = 0;
    for (gko::int32 i = 0; diag_lut[i + 1] <= n && i < max_b_s - 1; i++) {
        result = i;
    }
    gko::int32 tmp = n - diag_lut[result + 1];
    return tmp == 0 ? result : tmp - 1;
}

constexpr auto sub_block_lut = lut<max_nz_block + 1>(sub_block);

/// @brief equal to {0, 0, 1, 0, 1, 2, 0, 1, 2, 3, ...}
/// @param n storage scheme id of the subblock(/entry in a base block)
/// @return relative id of the diagonal entry above (/of the subblock itself)
gko::int32 precomputed_block(gko::int32 n) { return sub_block_lut[n]; }

/// @brief equal to {0, 2, 5, 9, 14, ...}
/// @param n relative id of the diagonal entry
/// @return id of the diagonal entry in the rowwise storage scheme
gko::int32 precomputed_diag(gko::int32 n) { return diag_lut[n + 1]; }

/// @brief equal to {0, 1, 3, 6, 9, ...}
/// @param n block size
/// @return number of nonzeros in a triangular block of given size
gko::int32 precomputed_nz_p_b(gko::int32 n) { return diag_lut[n] + 1; }

#define GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL(ValueType, IndexType)  \
    void simple_apply(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const IndexType* l_diag_rows, const ValueType* l_diag_vals,         \
        const IndexType* l_spmv_row_ptrs, const IndexType* l_spmv_col_idxs, \
        const ValueType* l_spmv_vals, const IndexType* permutation_idxs,    \
        const preconditioner::storage_scheme& storage_scheme,               \
        matrix::Dense<ValueType>* b_perm, matrix::Dense<ValueType>* x,      \
        int kernel_version)

#define GKO_DECLARE_GAUSS_SEIDEL_PREPERMUTED_SIMPLE_APPLY_KERNEL(ValueType, \
                                                                 IndexType) \
    void prepermuted_simple_apply(                                          \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const IndexType* l_diag_rows, const ValueType* l_diag_vals,         \
        const IndexType* l_spmv_row_ptrs, const IndexType* l_spmv_col_idxs, \
        const ValueType* l_spmv_vals,                                       \
        const preconditioner::storage_scheme& storage_scheme,               \
        const IndexType* permutation_idxs,                                  \
        const matrix::Dense<ValueType>* b_perm,                             \
        matrix::Dense<ValueType>* x_perm, int kernel_version)

#define GKO_DECLARE_GAUSS_SEIDEL_GET_COLORING_KERNEL(ValueType, IndexType) \
    void get_coloring(                                                     \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix, \
        array<IndexType>& vertex_colors, IndexType* max_color)

#define GKO_DECLARE_GAUSS_SEIDEL_GET_BLOCK_COLORING_KERNEL(ValueType,      \
                                                           IndexType)      \
    void get_block_coloring(                                               \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix, \
        const IndexType* block_ordering, const IndexType block_size,       \
        IndexType* vertex_colors, IndexType* max_color)

#define GKO_DECLARE_GAUSS_SEIDEL_GET_SECONDARY_ORDERING_KERNEL(IndexType)  \
    void get_secondary_ordering(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        IndexType* permutation_idxs,                                       \
        preconditioner::storage_scheme& storage_scheme,                    \
        const IndexType base_block_size, const IndexType lvl_2_block_size, \
        const IndexType* color_block_ptrs, const IndexType max_color,      \
        const bool use_padding)

#define GKO_DECLARE_GAUSS_SEIDEL_ASSIGN_TO_BLOCKS_KERNEL(ValueType, IndexType) \
    void assign_to_blocks(                                                     \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,     \
        IndexType* block_ordering, const IndexType* degrees, int8* visited,    \
        const IndexType block_size)

#define GKO_DECLARE_GAUSS_SEIDEL_GET_PERMUTATION_FROM_COLORING_KERNEL( \
    IndexType)                                                         \
    void get_permutation_from_coloring(                                \
        std::shared_ptr<const DefaultExecutor> exec,                   \
        const IndexType num_nodes, IndexType* coloring,                \
        const IndexType max_color, IndexType* color_ptrs,              \
        IndexType* permutation_idxs, const IndexType* block_ordering)

#define GKO_DECLARE_GAUSS_SEIDEL_GET_DEGREE_OF_NODES_KERNEL(IndexType)    \
    void get_degree_of_nodes(std::shared_ptr<const DefaultExecutor> exec, \
                             const IndexType num_vertices,                \
                             const IndexType* const row_ptrs,             \
                             IndexType* const degrees)

#define GKO_DECLARE_GAUSS_SEIDEL_SETUP_BLOCKS_KERNEL(ValueType, IndexType)    \
    void setup_blocks(std::shared_ptr<const DefaultExecutor> exec,            \
                      const matrix::Csr<ValueType, IndexType>* system_matrix, \
                      const IndexType* permutation_idxs,                      \
                      const IndexType* inv_permutation_idxs,                  \
                      preconditioner::storage_scheme& storage_scheme,         \
                      IndexType* l_diag_rows, IndexType* l_diag_mtx_col_idxs, \
                      ValueType* l_diag_vals, IndexType* l_spmv_row_ptrs,     \
                      IndexType* l_spmv_col_idxs,                             \
                      IndexType* l_spmv_mtx_col_idxs, ValueType* l_spmv_vals, \
                      IndexType* u_diag_rows, IndexType* u_diag_mtx_col_idxs, \
                      ValueType* u_diag_vals, IndexType* u_spmv_row_ptrs,     \
                      IndexType* u_spmv_col_idxs,                             \
                      IndexType* u_spmv_mtx_col_idxs, ValueType* u_spmv_vals)

#define GKO_DECLARE_GAUSS_SEIDEL_FILL_WITH_VALS_KERNEL(ValueType, IndexType) \
    void fill_with_vals(                                                     \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const matrix::Csr<ValueType, IndexType>* system_matrix,              \
        const IndexType* permutation_idxs,                                   \
        const preconditioner::storage_scheme& storage_scheme,                \
        const IndexType diag_num_elems, const IndexType* l_diag_rows,        \
        const IndexType* l_diag_mtx_col_idxs, ValueType* l_diag_vals,        \
        const IndexType* l_spmv_row_ptrs, const IndexType* l_spmv_col_idxs,  \
        const IndexType* l_spmv_mtx_col_idxs, ValueType* l_spmv_vals,        \
        const IndexType* u_diag_rows, const IndexType* u_diag_mtx_col_idxs,  \
        ValueType* u_diag_vals, const IndexType* u_spmv_row_ptrs,            \
        const IndexType* u_spmv_col_idxs,                                    \
        const IndexType* u_spmv_mtx_col_idxs, ValueType* u_spmv_vals)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                          \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_PREPERMUTED_SIMPLE_APPLY_KERNEL(ValueType,       \
                                                             IndexType);      \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_GET_COLORING_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_GET_BLOCK_COLORING_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_ASSIGN_TO_BLOCKS_KERNEL(ValueType, IndexType);   \
    template <typename IndexType>                                             \
    GKO_DECLARE_GAUSS_SEIDEL_GET_PERMUTATION_FROM_COLORING_KERNEL(IndexType); \
    template <typename IndexType>                                             \
    GKO_DECLARE_GAUSS_SEIDEL_GET_SECONDARY_ORDERING_KERNEL(IndexType);        \
    template <typename IndexType>                                             \
    GKO_DECLARE_GAUSS_SEIDEL_GET_DEGREE_OF_NODES_KERNEL(IndexType);           \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_SETUP_BLOCKS_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_FILL_WITH_VALS_KERNEL(ValueType, IndexType)

GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(gauss_seidel,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES

}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_PRECONDITIONER_GAUSS_SEIDEL_KERNELS_HPP_