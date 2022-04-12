#ifndef MULTIGRID_UTILS
#define MULTIGRID_UTILS

#include <chrono>
#include <ginkgo/ginkgo.hpp>

// FLOPs for CG solve without preconditioning
template <typename ValueType, typename IndexType>
int calculate_FLOPS(int num_CG_sets, int num_iters_per_CG_solve,
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
int calculate_FLOPS(
    int num_CG_sets, int num_iters_per_CG_solve,
    gko::matrix::Csr<ValueType, IndexType>* mat, int num_mg_levels,
    std::vector<std::shared_ptr<const gko::multigrid::MultigridLevel>>
        mg_level_list,
    int num_presmooth_steps, int num_postsmooth_steps)
{
    // similar to HPCG reference implementation
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    const auto total_iters = num_CG_sets * num_iters_per_CG_solve;

    const auto num_ops_cg =
        calculate_FLOPS(num_CG_sets, num_iters_per_CG_solve, mat);

    const auto num_ops_mg = 0;
    for (auto i = 0; i < num_mg_levels; i++) {
        // make 1 test run of the mg as a solver ->pass on the list of fine_ops
        // ->cast to csr ->get_num_stored_elems()
        const csr* op_at_level =
            dynamic_cast<const csr*>(mg_level_list.at(i)->get_fine_op().get());
        const auto nnz_at_level = op_at_level->get_num_stored_elements();
        const auto num_rows_at_level = op_at_level->get_size()[0];
        // num_ops_mg += num_presmooth_steps * total_iters *; //todo
        // num_ops_mg += num_postsmooth_steps * total_iters *;//todo
        // num_ops_mg += total_iters * ; //todo, fine grid residual calculation

        // TODO ops per prolong / restrict
        //  num_ops_mg += total_iters * num_rows_at_level * ;//prolong
        //  num_ops_mg += total_iters * num_rows_at_level * ;//restrict
    }
    // num_ops_mg += total_iters * ;//todo cost of the ir solve on the coarsest
    // level
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