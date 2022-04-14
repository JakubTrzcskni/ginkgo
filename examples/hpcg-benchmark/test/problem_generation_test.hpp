#ifndef MATRIX_GENERATION_TEST
#define MATRIX_GENERATION_TEST

#include <array>
#include <iostream>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/ginkgo.hpp>
#include "examples/hpcg-benchmark/include/geometric-multigrid.hpp"

template <typename ValueType>
void compare_dense_matrices(std::shared_ptr<const gko::Executor> exec,
                            gko::matrix::Dense<ValueType>* device_mat,
                            const gko::matrix::Dense<ValueType>* host_mat)
{
    using array = gko::Array<ValueType>;
    auto host_exec = exec->get_master();

    const auto values_host =
        array::const_view(host_exec, host_mat->get_num_stored_elements(),
                          host_mat->get_const_values());
    auto values_device = array::view(exec, host_mat->get_num_stored_elements(),
                                     device_mat->get_values());
    values_device.set_executor(host_exec);

    auto diff_on_values = 0;
    for (auto i = 0; i < values_host.get_num_elems(); i++) {
        diff_on_values +=
            values_host.get_const_data()[i] - values_device.get_const_data()[i];
    }
    if (diff_on_values == 0) {
        std::cout << "passed"
                  << "\n";
    } else {
        std::cout << "diff on values\n" << diff_on_values << "\n";
    }
}


template <typename ValueType, typename IndexType>
void compare_csr_matrices(
    std::shared_ptr<const gko::Executor> exec,
    gko::matrix::Csr<ValueType, IndexType>* device_mat,
    const gko::matrix::Csr<ValueType, IndexType>* host_mat)
{
    using id_array = gko::Array<IndexType>;
    using val_array = gko::Array<ValueType>;
    auto host_exec = exec->get_master();

    const auto row_ptrs_host = id_array::const_view(
        host_exec, host_mat->get_size()[0] + 1, host_mat->get_const_row_ptrs());

    auto row_ptrs_device = id_array::view(exec, device_mat->get_size()[0] + 1,
                                          device_mat->get_row_ptrs());
    row_ptrs_device.set_executor(host_exec);

    auto diff_on_row_ptrs = 0;
    for (auto i = 0; i < row_ptrs_host.get_num_elems(); i++) {
        diff_on_row_ptrs += row_ptrs_host.get_const_data()[i] -
                            row_ptrs_device.get_const_data()[i];
    }

    const auto col_idxs_host =
        id_array::const_view(host_exec, host_mat->get_num_stored_elements(),
                             host_mat->get_const_col_idxs());

    auto col_idxs_device =
        id_array::view(exec, device_mat->get_num_stored_elements(),
                       device_mat->get_col_idxs());
    col_idxs_device.set_executor(host_exec);

    auto diff_on_col_idxs = 0;
    for (auto i = 0; i < col_idxs_host.get_num_elems(); i++) {
        diff_on_col_idxs += col_idxs_host.get_const_data()[i] -
                            col_idxs_device.get_const_data()[i];
    }


    const auto values_host =
        val_array::const_view(host_exec, host_mat->get_num_stored_elements(),
                              host_mat->get_const_values());

    auto values_device = val_array::view(
        exec, device_mat->get_num_stored_elements(), device_mat->get_values());
    values_device.set_executor(host_exec);

    auto diff_on_values = 0;
    for (auto i = 0; i < values_host.get_num_elems(); i++) {
        diff_on_values +=
            values_host.get_const_data()[i] - values_device.get_const_data()[i];
    }

    if (diff_on_col_idxs + diff_on_row_ptrs + diff_on_values == 0) {
        std::cout << "passed"
                  << "\n";
    } else {
        std::cout << "diff on values\n" << diff_on_values << "\n";
        std::cout << "diff on col idxs\n" << diff_on_col_idxs << "\n";
        std::cout << "diff on row ptrs\n" << diff_on_row_ptrs << "\n";
    }
}

template <typename ValueType, typename IndexType>
void test_matrix_generation(std::shared_ptr<const gko::Executor> exec,
                            const gko::multigrid::problem_geometry& geometry,
                            ValueType value_help, IndexType index_help)
{
    using csr = gko::matrix::Csr<ValueType, IndexType>;

    const auto mat_size = get_dp_3D(geometry);
    auto system_matrix = share(csr::create(exec, gko::dim<2>(mat_size)));
    matrix_generation_kernel(exec, geometry.nx, geometry.ny, geometry.nz,
                             lend(system_matrix));

    auto host_system_matrix =
        share(csr::create(exec->get_master(), system_matrix->get_size()));

    generate_problem_matrix(exec->get_master(), geometry,
                            lend(host_system_matrix));
    std::cout << "system matrix device/host" << std::endl;

    compare_csr_matrices(exec, lend(system_matrix), lend(host_system_matrix));
}

template <typename ValueType, typename IndexType>
void test_rhs_and_x_generation(std::shared_ptr<const gko::Executor> exec,
                               gko::multigrid::problem_geometry& geometry,
                               ValueType value_help, IndexType index_help)
{
    using vec = gko::matrix::Dense<ValueType>;
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    auto host_exec = exec->get_master();

    const auto mat_size = get_dp_3D(geometry);
    auto system_matrix = csr::create(exec, gko::dim<2>(mat_size));
    auto rhs = vec::create(exec, gko::dim<2>(mat_size, 1));
    auto x_exact = vec::create(exec, gko::dim<2>(mat_size, 1));
    auto x = vec::create(exec, gko::dim<2>(mat_size, 1));

    generate_problem(exec, geometry, lend(system_matrix), lend(rhs), lend(x),
                     lend(x_exact));

    auto host_matrix = share(csr::create(host_exec, system_matrix->get_size()));
    auto host_rhs = share(
        vec::create(host_exec, gko::dim<2>(system_matrix->get_size()[0], 1)));
    auto host_x_exact = share(
        vec::create(host_exec, gko::dim<2>(system_matrix->get_size()[0], 1)));
    auto host_x = share(
        vec::create(host_exec, gko::dim<2>(system_matrix->get_size()[0], 1)));

    generate_problem(host_exec, geometry, lend(host_matrix), lend(host_rhs),
                     lend(host_x), lend(host_x_exact));

    std::cout << "rhs device/host" << std::endl;
    compare_dense_matrices(exec, lend(rhs), lend(host_rhs));
    std::cout << "x_exact device/host" << std::endl;
    compare_dense_matrices(exec, lend(x_exact), lend(host_x_exact));
    std::cout << "x device/host" << std::endl;
    compare_dense_matrices(exec, lend(x), lend(host_x));
}

#endif