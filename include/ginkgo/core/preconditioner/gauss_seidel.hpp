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

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_GAUSS_SEIDEL_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_GAUSS_SEIDEL_HPP_

#include <vector>

#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>

#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


namespace gko {
namespace preconditioner {


struct general_block {
    general_block() = default;
    general_block(int32 start_val_storage_id, int32 start_row_global,
                  int32 end_row_global)
        : val_storage_id_{start_val_storage_id},
          start_row_global_{start_row_global},
          end_row_global_{end_row_global}
    {}
    int32 val_storage_id_ = 0;
    int32 start_row_global_ = 0;
    int32 end_row_global_ = 0;
};


struct storage_scheme {
    storage_scheme() = default;
    storage_scheme(int32 num_blocks, bool symm = false)
        : num_blocks_{num_blocks}, symm_{symm}
    {
        forward_solve_.reserve(num_blocks);
        if (symm) {
            backward_solve_.reserve(num_blocks);
        }
    }
    int32 num_blocks_;
    bool symm_;
    std::vector<std::shared_ptr<general_block>> forward_solve_;
    std::vector<std::shared_ptr<general_block>> backward_solve_;
};

struct spmv_block : general_block {
    spmv_block(int32 start_row_global, int32 start_col_global,
               int32 end_col_global)
        : general_block(0, start_row_global, 0),
          start_col_global_{start_col_global},
          end_col_global_{end_col_global}
    {}
    spmv_block(int32 start_row_ptrs_storage_id, int32 start_val_storage_id,
               int32 start_row_global, int32 end_row_global,
               int32 start_col_global, int32 end_col_global)
        : general_block(start_val_storage_id, start_row_global, end_row_global),
          row_ptrs_storage_id_{start_row_ptrs_storage_id},
          start_col_global_{start_col_global},
          end_col_global_{end_col_global}
    {}
    void update(int32 start_row_ptrs_storage_id, int32 start_val_storage_id,
                int32 start_row_global, int32 end_row_global,
                int32 start_col_global, int32 end_col_global)
    {
        row_ptrs_storage_id_ = start_row_ptrs_storage_id;
        val_storage_id_ = start_val_storage_id;
        start_row_global_ = start_row_global;
        start_col_global_ = start_col_global;
        end_row_global_ = end_row_global;
        end_col_global_ = end_col_global;
    }
    int32 row_ptrs_storage_id_ = 0;
    int32 start_col_global_ = 0;
    int32 end_col_global_ = 0;
};

struct parallel_block : general_block {
    parallel_block() = default;
    parallel_block(int32 start_val_storage_id, int32 start_row_global,
                   int32 end_row_global, int32 degree_of_parallelism,
                   int32 base_block_size, int32 lvl_2_block_size, bool residual)
        : general_block(start_val_storage_id, start_row_global, end_row_global),
          degree_of_parallelism_{degree_of_parallelism},
          base_block_size_{base_block_size},
          lvl_2_block_size_{lvl_2_block_size},
          residual_{residual}
    {
        parallel_blocks_.reserve(degree_of_parallelism);
        lvl_1_block_size_ = base_block_size * lvl_2_block_size;
        nz_p_b_block_ =
            (base_block_size * base_block_size - base_block_size) / 2 +
            base_block_size;
    }
    int32 degree_of_parallelism_;
    int32 base_block_size_ = 4;
    int32 nz_p_b_block_ = 10;
    int32 lvl_2_block_size_ = 4;
    int32 lvl_1_block_size_ = 16;
    bool residual_;  // true if there are trailing base blocks, false if current
                     // color consists only of lvl 1 blocks
    std::vector<std::shared_ptr<general_block>> parallel_blocks_;
};

// template <int base_block_size, int lvl_2_block_size>
struct lvl_1_block : general_block {
    lvl_1_block(int32 start_val_storage_id, int32 start_row_global,
                int32 end_row_global)
        : general_block(start_val_storage_id, start_row_global, end_row_global)
    {}
};

// template <int base_block_size>
struct base_block_aggregation : general_block {
    base_block_aggregation(int32 start_val_storage_id, int32 start_row_global,
                           int32 end_row_global, int32 num_base_blocks)
        : general_block(start_val_storage_id, start_row_global, end_row_global),
          num_base_blocks_{num_base_blocks}
    {}
    int32 num_base_blocks_;
};

template <typename ValueType = default_precision, typename IndexType = int32>
class GaussSeidel : public EnableLinOp<GaussSeidel<ValueType, IndexType>>,
                    // public ConvertibleTo<matrix::Dense<ValueType>>,
                    // public WritableToMatrixData<ValueType, IndexType>,
                    public Transposable {
    friend class EnableLinOp<GaussSeidel>;
    friend class EnablePolymorphicObject<GaussSeidel, LinOp>;
    // friend class

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Csr = matrix::Csr<ValueType, IndexType>;
    using Dense = matrix::Dense<ValueType>;
    using LTrs = solver::LowerTrs<value_type, index_type>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    GaussSeidel& operator=(const GaussSeidel& other);

    GaussSeidel& operator=(GaussSeidel&& other);

    GaussSeidel(const GaussSeidel& other);

    GaussSeidel(GaussSeidel&& other);

    array<index_type> get_vertex_colors() { return vertex_colors_; }

    array<index_type> get_permutation_idxs() { return permutation_idxs_; }

    array<index_type> get_color_ptrs() { return color_ptrs_; }

    std::shared_ptr<Csr> get_ltr_matrix() { return lower_triangular_matrix_; }

    std::shared_ptr<Csr> get_utr_matrix() { return upper_triangular_matrix_; }

    std::vector<index_type> get_level_ptrs() { return level_ptrs_; }

    void update_system(value_type* values);

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, true);

        bool GKO_FACTORY_PARAMETER_SCALAR(use_coloring, true);

        // hierarchical algebraic block coloring strategy
        bool GKO_FACTORY_PARAMETER_SCALAR(use_HBMC, false);

        // TODO change to uint
        size_t GKO_FACTORY_PARAMETER_SCALAR(base_block_size, 4);

        size_t GKO_FACTORY_PARAMETER_SCALAR(lvl2_block_size, 32);

        // determines if ginkgo lower triangular solver should be used
        // if reference solver is used no coloring&reordering will take place
        bool GKO_FACTORY_PARAMETER_SCALAR(use_reference, false);

        // determines if GS/SOR or SGS/SSOR should be used
        bool GKO_FACTORY_PARAMETER_SCALAR(symmetric_preconditioner, false);

        // relevant only for SOR/SSOR - has to be between 0.0 and 2.0
        double GKO_FACTORY_PARAMETER_SCALAR(relaxation_factor, 1.0);
    };
    GKO_ENABLE_LIN_OP_FACTORY(GaussSeidel, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    // empty GS preconditioner
    explicit GaussSeidel(std::shared_ptr<const Executor> exec)
        : EnableLinOp<GaussSeidel>(exec),
          relaxation_factor_{parameters_.relaxation_factor},
          symmetric_preconditioner_{parameters_.symmetric_preconditioner},
          use_reference_{parameters_.use_reference},
          use_coloring_{parameters_.use_coloring}
    {}

    // GS preconditioner from a factory
    explicit GaussSeidel(const Factory* factory,
                         std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<GaussSeidel>(factory->get_executor(),
                                   gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          lower_triangular_matrix_{Csr::create(factory->get_executor())},
          upper_triangular_matrix_{Csr::create(factory->get_executor())},
          lower_trs_factory_{share(LTrs::build().on(factory->get_executor()))},
          vertex_colors_{array<index_type>(factory->get_executor(),
                                           system_matrix->get_size()[0])},
          color_ptrs_{array<index_type>(factory->get_executor())},
          permutation_idxs_{array<index_type>(factory->get_executor())},
          inv_permutation_idxs_{array<index_type>(factory->get_executor())},
          base_block_size_{parameters_.base_block_size},
          lvl2_block_size_{parameters_.lvl2_block_size},
          relaxation_factor_{parameters_.relaxation_factor},
          symmetric_preconditioner_{parameters_.symmetric_preconditioner},
          use_reference_{parameters_.use_reference},
          use_coloring_{parameters_.use_coloring},
          l_diag_rows_{array<index_type>(factory->get_executor())},
          l_diag_mtx_col_idxs_{array<index_type>(factory->get_executor())},
          l_diag_vals_{array<value_type>(factory->get_executor())},
          l_spmv_row_ptrs_{array<index_type>(factory->get_executor())},
          l_spmv_col_idxs_{array<index_type>(factory->get_executor())},
          l_spmv_mtx_col_idxs_{array<index_type>(factory->get_executor())},
          l_spmv_vals_{array<value_type>(factory->get_executor())}
    {
        if (parameters_.use_HBMC == true) {
            if (parameters_.symmetric_preconditioner) {
                u_diag_rows_ = array<index_type>(factory->get_executor());
                u_diag_mtx_col_idxs_ =
                    array<index_type>(factory->get_executor());
                u_diag_vals_ = array<value_type>(factory->get_executor());
                u_spmv_row_ptrs_ = array<index_type>(factory->get_executor());
                u_spmv_col_idxs_ = array<index_type>(factory->get_executor());
                u_spmv_mtx_col_idxs_ =
                    array<index_type>(factory->get_executor());
                u_spmv_vals_ = array<value_type>(factory->get_executor());
            }
            this->generate_HBMC(system_matrix, parameters_.skip_sorting);

        } else {
            this->generate(system_matrix, parameters_.skip_sorting);
        }
    }

    void generate(std::shared_ptr<const LinOp> system_matrix,
                  bool skip_sorting);

    void generate_HBMC(std::shared_ptr<const LinOp> system_matrix,
                       bool skip_sorting);

    std::unique_ptr<matrix::SparsityCsr<value_type, index_type>>
    get_adjacency_matrix(matrix_data<value_type, index_type>& mat_data,
                         bool is_symmetric = false);

    index_type get_coloring(matrix_data<value_type, index_type>& mat_data,
                            bool is_symmetric = false);

    void initialize_blocks();

    void reserve_mem_for_block_structure(
        const matrix::SparsityCsr<value_type, index_type>* system_matrix,
        const index_type num_base_blocks, const index_type base_block_size,
        const index_type num_colors);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    array<index_type> generate_block_structure(
        const matrix::SparsityCsr<value_type, index_type>* adjacency_matrix,
        const index_type block_size, const index_type lvl_2_block_size = 0);


private:
    std::shared_ptr<Csr>
        lower_triangular_matrix_{};  // aka matrix used in the preconditioner
    std::shared_ptr<Csr> upper_triangular_matrix_{};
    std::shared_ptr<const LinOp> lower_trs_{};
    std::shared_ptr<typename LTrs::Factory> lower_trs_factory_{};
    std::shared_ptr<const LinOp> upper_trs_{};
    array<index_type> vertex_colors_;
    array<index_type> color_ptrs_;  // assuming that the colors found constitute
                                    // a span [0,max_color], i.e., there are no
                                    // gaps in the assigned color numbers
    array<index_type> permutation_idxs_;
    array<index_type> inv_permutation_idxs_;
    std::vector<std::unique_ptr<LinOp>> block_ptrs_;
    std::vector<index_type> level_ptrs_;
    size_t base_block_size_;
    size_t lvl2_block_size_;
    double relaxation_factor_;
    bool symmetric_preconditioner_;
    bool use_reference_;
    bool use_coloring_;
    storage_scheme hbmc_storage_scheme_{};
    array<index_type> l_diag_rows_;
    array<index_type> l_diag_mtx_col_idxs_;
    array<value_type> l_diag_vals_;
    array<index_type> l_spmv_row_ptrs_;
    array<index_type> l_spmv_col_idxs_;
    array<index_type> l_spmv_mtx_col_idxs_;
    array<value_type> l_spmv_vals_;
    array<index_type> u_diag_rows_;
    array<value_type> u_diag_vals_;
    array<index_type> u_diag_mtx_col_idxs_;
    array<index_type> u_spmv_row_ptrs_;
    array<index_type> u_spmv_col_idxs_;
    array<index_type> u_spmv_mtx_col_idxs_;
    array<value_type> u_spmv_vals_;
};
}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_GAUSS_SEIDEL_HPP_