#ifndef MULTIGRID_UTILS
#define MULTIGRID_UTILS

#include <ginkgo/ginkgo.hpp>
#include "geometric-multigrid.hpp"

static int get_dp_3D(const int nx, const int ny, const int nz)
{
    return (nx + 1) * (ny + 1) * (nz + 1);
}
static int get_dp_3D(const gko::multigrid::problem_geometry& geometry)
{
    return get_dp_3D(geometry.nx, geometry.ny, geometry.nz);
}
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
    std::shared_ptr<const gko::Executor> exec, const int discretization_points,
    gko::matrix::Dense<ValueType>* u_device,
    const gko::matrix::Dense<ValueType>* u_exact)
{
    using array = gko::Array<ValueType>;
    auto values_device =
        array::view(exec, discretization_points, u_device->get_values());
    values_device.set_executor(exec->get_master());
    const ValueType h = 1.0 / static_cast<ValueType>(discretization_points + 1);
    gko::remove_complex<ValueType> error = 0.0;
    for (int i = 0; i < discretization_points; ++i) {
        using std::abs;
        error += abs(values_device.get_const_data()[i] -
                     u_exact->get_const_values()[i]) /
                 abs(u_exact->get_const_values()[i]);
    }
    return error;
}
#endif