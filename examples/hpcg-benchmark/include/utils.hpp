#include <ginkgo/ginkgo.hpp>
#include "geometric-multigrid.hpp"

int get_dp_3D(gko::multigrid::problem_geometry& geometry)
{
    return (geometry.nx + 1) * (geometry.ny + 1) * (geometry.nz + 1);
}
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