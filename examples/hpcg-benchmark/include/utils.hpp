#ifndef MULTIGRID_UTILS
#define MULTIGRID_UTILS

#include <chrono>
#include <ginkgo/ginkgo.hpp>

template <typename ValueType, typename IndexType>
size_t calculate_bandwidth(size_t num_CG_sets, size_t num_iters_per_CG_solve,
                           gko::matrix::Csr<ValueType, IndexType>* mat)
{
    // ginkgo cg solver:
    /* Memory movement summary:
     * 18n * values + matrix/preconditioner storage
     * 1x SpMV:           2n * values + storage
     * 1x Preconditioner: 2n * values + storage
     * 2x dot             4n
     * 1x step 1 (axpy)   3n
     * 1x step 2 (axpys)  6n
     * 1x norm2 residual   n
     */
    const auto nnz = mat->get_num_stored_elements();
    const auto num_rows = mat->get_size()[0];
    const auto total_iters = num_CG_sets * num_iters_per_CG_solve;

    const auto num_reads_dot =
        2 * total_iters * 2 * num_rows * sizeof(ValueType);
    const auto num_writes_dot =
        2 * total_iters * 1 * num_rows * sizeof(ValueType);
    const auto num_reads_axpby =
        3 * total_iters * 2 * num_rows * sizeof(ValueType);
    const auto num_writes_axpby =
        3 * total_iters * 1 * num_rows * sizeof(ValueType);
    const auto num_reads_spmv = (total_iters + num_CG_sets) *
                                (nnz * (sizeof(ValueType) + sizeof(IndexType)) +
                                 num_rows * sizeof(ValueType));
    const auto num_writes_spmv =
        (total_iters + num_CG_sets) * num_rows * sizeof(ValueType);

    const auto num_reads_norm2 = total_iters * 1 * num_rows * sizeof(ValueType);
    const auto num_writes_norm2 = total_iters * 1;

    return num_reads_dot + num_writes_dot + num_reads_axpby + num_writes_axpby +
           num_reads_spmv + num_writes_spmv + num_reads_norm2 +
           num_writes_norm2;
};


// FLOPs for CG solve without preconditioning
template <typename ValueType, typename IndexType>
size_t calculate_FLOPS(size_t num_CG_sets, size_t num_iters_per_CG_solve,
                       gko::matrix::Csr<ValueType, IndexType>* mat)
{
    const auto nnz = mat->get_num_stored_elements();
    const auto num_rows = mat->get_size()[0];
    const auto total_iters = num_CG_sets * num_iters_per_CG_solve;

    const auto num_ops_dot = 2 * total_iters * 2 * num_rows;
    const auto num_ops_axpby = 3 * total_iters * 2 * num_rows;
    const auto num_ops_spmv = (total_iters + num_CG_sets) * 2 * nnz;
    const auto num_ops_norm2 = total_iters * 2 * num_rows;

    return num_ops_dot + num_ops_axpby + num_ops_spmv + num_ops_norm2;
};

// FLOPs for CG solve with MG-preconditioner
template <typename ValueType, typename IndexType>
size_t calculate_FLOPS(
    size_t num_CG_sets, size_t num_iters_per_CG_solve,
    gko::matrix::Csr<ValueType, IndexType>* mat,
    std::vector<std::shared_ptr<const gko::multigrid::MultigridLevel>>
        mg_level_list,
    size_t num_presmooth_steps, size_t num_postsmooth_steps,
    bool full_weighting)
{
    // similar to HPCG reference implementation
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    const auto total_iters = num_CG_sets * num_iters_per_CG_solve;

    const auto num_ops_cg =
        calculate_FLOPS(num_CG_sets, num_iters_per_CG_solve, mat);

    size_t num_mg_levels = mg_level_list.size();
    size_t num_ops_mg = 0;
    for (auto i = 0; i < num_mg_levels; i++) {
        const csr* op_at_level =
            dynamic_cast<const csr*>(mg_level_list.at(i)->get_fine_op().get());
        const auto nnz_at_level = op_at_level->get_num_stored_elements();
        const auto num_rows_at_level = op_at_level->get_size()[0];
        num_ops_mg +=
            (num_presmooth_steps + num_postsmooth_steps) * total_iters *
            (2 * nnz_at_level  // iterative refinement -> 1x spmv per smoothing
                               // step
             + 2 * num_rows_at_level);  // application of the bj smoother -> 1x
                                        // axpy per smoothing step

        num_ops_mg +=
            total_iters * 2 *  // ?
            nnz_at_level;      // fine grid residual calculation -> 1x spmv

        auto curr_dp_1D = static_cast<int>(cbrt(num_rows_at_level)) - 1;
        // if prolongation and restriction use full weighting
        // works for symmetric grids (dp_x = dp_y = dp_z)
        if (full_weighting && i != 0) {  // prolong
            // in case of prolongation curr_dp_1D is the 1D dimension of the
            // coarse grid
            //  3 for fine grid points on the coarse grid
            //  3 + 1 for fine grid points with 2 out of 3 coord. on the coarse
            //  grid
            //  3 + 3 for fine grid points with 1 out of 3 coord. on the
            //  coarse grid 3 + 7 for fine grid points off the coarse grid
            auto fine_dp_1D = 2 * curr_dp_1D;
            auto num_coarse_grid_points = num_rows_at_level;
            auto num_fine_grid_points =
                (fine_dp_1D + 1) * (fine_dp_1D + 1) * (fine_dp_1D + 1);
            auto num_fine_points_off_coarse_grid =
                curr_dp_1D * curr_dp_1D * curr_dp_1D;
            auto num_fine_points_2_out_of_3_dim_on_coarse =
                3 * curr_dp_1D * (curr_dp_1D + 1) * (curr_dp_1D + 1);
            auto num_fine_points_1_out_of_3_dim_on_coarse =
                num_fine_grid_points - num_coarse_grid_points -
                num_fine_points_off_coarse_grid -
                num_fine_points_2_out_of_3_dim_on_coarse;
            num_ops_mg +=
                total_iters * (3 * num_coarse_grid_points +
                               4 * num_fine_points_2_out_of_3_dim_on_coarse +
                               6 * num_fine_points_1_out_of_3_dim_on_coarse +
                               10 * num_fine_points_off_coarse_grid);


            // alternative: create explicit prolong -> get nnz -> application is
            // a standard spmv
        }
        if (full_weighting && i != num_mg_levels - 1) {  // restrict
            // nnz on border * 1 + nnz inside * 27 * (1 + 3)
            auto fine_dp_1D = curr_dp_1D;
            auto coarse_dp_1D = fine_dp_1D / 2;
            auto points_inside =
                (coarse_dp_1D - 1) * (coarse_dp_1D - 1) * (coarse_dp_1D - 1);
            auto points_on_border = nnz_at_level - points_inside;
            num_ops_mg +=
                total_iters * (points_on_border + points_inside * 27 * 4);
        }
        if (i == num_mg_levels - 1) {      // higher lvl -> coarser
            num_ops_mg += total_iters * 4  // num iters of the coarsest solver
                          * (2 * nnz_at_level +
                             2 * num_rows_at_level);  // same as smoother
        }
    }
    return num_ops_cg + num_ops_mg;
};

static int calc_nnz(const int nx, const int ny, const int nz)
{
    return (nx + 1) * (ny + 1) * (nz + 1) * 27 + 64 + ((nx + 1) - 2) * 4 * 3 +
           ((ny + 1) - 2) * 4 * 3 + ((nz + 1) - 2) * 4 * 3 -
           2 * (nx + 1) * (ny + 1) * 9 - 2 * (nx + 1) * (nz + 1) * 9 -
           2 * (ny + 1) * (nz + 1) * 9;
};

// Prints the solution `u`.
template <typename ValueType>
void print_solution(ValueType u0, ValueType u1,
                    const gko::matrix::Dense<ValueType>* u)
{
    // std::cout << u0 << '\n';
    for (int i = 0; i < u->get_size()[0]; ++i) {
        std::cout << u->get_const_values()[i] << '\n';
    }
    // std::cout << u1 << std::endl;
}
template <typename ValueType>
void print_matrix(const gko::matrix::Dense<ValueType>* u)
{
    for (int i = 0; i < u->get_size()[0]; ++i) {
        std::cout << u->get_const_values()[i] << '\n';
    }
}

// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename ValueType>
gko::remove_complex<ValueType> calculate_error(
    int discretization_points, const gko::matrix::Dense<ValueType>* u,
    const gko::matrix::Dense<ValueType>* u_exact)
{
    const ValueType h = 1.0 / static_cast<ValueType>(discretization_points + 1);
    gko::remove_complex<ValueType> error = 0.0;
    for (int i = 0; i < discretization_points; ++i) {
        using std::abs;
        error +=
            abs(u->get_const_values()[i] - u_exact->get_const_values()[i]) /
            abs(u_exact->get_const_values()[i]);
    }
    return error;
}


template <typename ValueType>
gko::remove_complex<ValueType> calculate_error_device(
    const int discretization_points, gko::matrix::Dense<ValueType>* u_device,
    gko::matrix::Dense<ValueType>* u_exact)
{
    using array = gko::Array<ValueType>;
    auto exec = u_exact->get_executor();
    auto values_device =
        array::view(u_device->get_executor(), discretization_points,
                    u_device->get_values());
    values_device.set_executor(exec->get_master());

    auto values_exact =
        array::view(exec, discretization_points, u_exact->get_values());
    values_exact.set_executor(exec->get_master());

    const ValueType h = 1.0 / static_cast<ValueType>(discretization_points + 1);
    gko::remove_complex<ValueType> error = 0.0;
    for (int i = 0; i < discretization_points; ++i) {
        using std::abs;
        error += abs(values_device.get_const_data()[i] -
                     values_exact.get_const_data()[i]) /
                 abs(values_exact.get_const_data()[i]);
    }
    return error;
}
#endif