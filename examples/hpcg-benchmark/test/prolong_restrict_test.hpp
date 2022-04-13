#include <ginkgo/ginkgo.hpp>
#include <random>
#include "examples/hpcg-benchmark/include/geometric-multigrid.hpp"

using namespace gko;
using namespace gko::multigrid;

template <typename ValueType, typename IndexType>
void test_restriction(std::shared_ptr<const gko::Executor> exec,
                      problem_geometry& geometry, ValueType value_help,
                      IndexType index_help)
{
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mgR = gko::multigrid::gmg_restriction<ValueType, IndexType>;
    using vec = gko::matrix::Dense<ValueType>;

    static std::default_random_engine e;
    static std::uniform_real_distribution<> dist(0., 1.);

    const unsigned int dp_3D = get_dp_3D(geometry);
    const auto c_nx = geometry.nx / 2;
    const auto c_ny = geometry.ny / 2;
    const auto c_nz = geometry.nz / 2;
    const unsigned int coarse_dp_3D = (c_nx + 1) * (c_ny + 1) * (c_nz + 1);

    ValueType coeffs[3] = {0.25, 0.5, 0.25};

    auto rhs_fine = vec::create(exec->get_master(), gko::dim<2>(dp_3D, 1));
    auto rhs_fine_device = vec::create(exec, gko::dim<2>(dp_3D, 1));
    for (auto i = 0; i < dp_3D; i++) {
        rhs_fine->at(i, 0) = dist(e);  // 1.0;
    }
    rhs_fine_device->copy_from(lend(rhs_fine));

    auto x_coarse_device = vec::create(exec, gko::dim<2>(coarse_dp_3D, 1));
    auto x_coarse_ref =
        vec::create(exec->get_master(), gko::dim<2>(coarse_dp_3D, 1));
    for (auto i = 0; i < coarse_dp_3D; i++) {
        x_coarse_ref->at(i, 0) = 0.0;
    }
    x_coarse_device->copy_from(lend(x_coarse_ref));

    auto restriction = mgR::create(exec, c_nx, c_ny, c_nz, coarse_dp_3D, dp_3D,
                                   coeffs[0], coeffs[1], coeffs[2]);
    auto restrict_explicit =
        share(csr::create(exec->get_master(), dim<2>(coarse_dp_3D, dp_3D)));
    create_explicit_restriction(lend(restrict_explicit), c_nx, c_ny, c_nz,
                                coeffs);
    restriction->apply(lend(rhs_fine), lend(x_coarse_device));
    exec->synchronize();
    restrict_explicit->apply(lend(rhs_fine), lend(x_coarse_ref));

    std::cout << "\nerror on restrict explicit/implicit\n"
              << calculate_error_device(coarse_dp_3D, lend(x_coarse_device),
                                        lend(x_coarse_ref))
              << std::endl;
}

template <typename ValueType, typename IndexType>
void test_prolongation(std::shared_ptr<const gko::Executor> exec,
                       problem_geometry& geometry, ValueType value_help,
                       IndexType index_help)
{
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mgP = gko::multigrid::gmg_prolongation<ValueType, IndexType>;
    using vec = gko::matrix::Dense<ValueType>;

    static std::default_random_engine e;
    static std::uniform_real_distribution<> dist(0., 1.);

    const auto c_nx = geometry.nx / 2;
    const auto c_ny = geometry.ny / 2;
    const auto c_nz = geometry.nz / 2;
    const unsigned int fine_dp_3D = get_dp_3D(geometry);
    const unsigned int dp_3D = get_dp_3D(c_nx, c_ny, c_nz);

    const ValueType coeffs[3] = {0.5, 1.0, 0.5};

    auto rhs_coarse_device = vec::create(exec, gko::dim<2>(dp_3D, 1));
    auto rhs_coarse_ref =
        vec::create(exec->get_master(), gko::dim<2>(dp_3D, 1));
    for (auto i = 0; i < dp_3D; i++) {
        rhs_coarse_ref->at(i, 0) = dist(e);
    }
    rhs_coarse_device->copy_from(lend(rhs_coarse_ref));

    auto x_fine_device = vec::create(exec, gko::dim<2>(fine_dp_3D, 1));
    auto x_fine_ref =
        vec::create(exec->get_master(), gko::dim<2>(fine_dp_3D, 1));
    for (auto i = 0; i < fine_dp_3D; i++) {
        x_fine_ref->at(i, 0) = 0.0;
    }
    x_fine_device->copy_from(lend(x_fine_ref));

    auto prolongation = mgP::create(exec, c_nx, c_ny, c_nz, dp_3D, fine_dp_3D,
                                    coeffs[0], coeffs[1], coeffs[2]);
    auto prolong_explicit =
        share(csr::create(exec->get_master(), gko::dim<2>(fine_dp_3D, dp_3D)));
    create_explicit_prolongation(lend(prolong_explicit), c_nx, c_ny, c_nz,
                                 coeffs);
    prolongation->apply(lend(rhs_coarse_device), lend(x_fine_device));
    exec->synchronize();
    prolong_explicit->apply(lend(rhs_coarse_ref), lend(x_fine_ref));

    std::cout << "error on prolong explicit/implicit\n"
              << calculate_error_device(fine_dp_3D, lend(x_fine_device),
                                        lend(x_fine_ref))
              << std::endl;
    // write(std::cout, lend(prolong_explicit));
}
