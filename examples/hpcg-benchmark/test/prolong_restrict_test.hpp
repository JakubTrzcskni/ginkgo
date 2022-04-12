#include <ginkgo/ginkgo.hpp>
#include <random>
#include "examples/hpcg-benchmark/include/geometric-multigrid.hpp"

using namespace gko;
using namespace gko::multigrid;

template <typename ValueType, typename IndexType>
void create_explicit_prolongation(gko::matrix::Csr<ValueType, IndexType>* mat,
                                  const int nx, const int ny, const int nz,
                                  const ValueType* coeffs)
{
    const auto coarse_dp = mat->get_size()[1];
    const auto fine_dp = mat->get_size()[0];
    GKO_ASSERT((nx + 1) * (ny + 1) * (nz + 1) == coarse_dp);

    GKO_ASSERT((2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1) == fine_dp);
    matrix_data<ValueType, IndexType> data{dim<2>(fine_dp, coarse_dp), {}};

#pragma omp parallel for
    for (auto c_z = 0; c_z <= nz; c_z++) {
        const auto f_z = 2 * c_z;
        for (auto c_y = 0; c_y <= ny; c_y++) {
            const auto f_y = 2 * c_y;
            for (auto c_x = 0; c_x <= nx; c_x++) {
                const auto f_x = 2 * c_x;
                const auto c_row = grid2index(c_x, c_y, c_z, nx, ny);
                const auto f_row = grid2index(f_x, f_y, f_z, 2 * nx, 2 * ny);
                for (auto ofs_z : {-1, 0, 1}) {
                    if (f_z + ofs_z > -1 && f_z + ofs_z <= 2 * nz) {
                        for (auto ofs_y : {-1, 0, 1}) {
                            if (f_y + ofs_y > -1 && f_y + ofs_y <= 2 * ny) {
                                for (auto ofs_x : {-1, 0, 1}) {
                                    if (f_x + ofs_x > -1 &&
                                        f_x + ofs_x <= 2 * nx) {
                                        const auto f_id =
                                            grid2index(ofs_x, ofs_y, ofs_z,
                                                       2 * nx, 2 * ny, f_row);

                                        const auto tmp = coeffs[ofs_z + 1] *
                                                         coeffs[ofs_y + 1] *
                                                         coeffs[ofs_x + 1];
                                        data.nonzeros.emplace_back(f_id, c_row,
                                                                   tmp);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    data.ensure_row_major_order();
    mat->read(data);
}

template <typename ValueType, typename IndexType>
void create_explicit_restriction(gko::matrix::Csr<ValueType, IndexType>* mat,
                                 int nx, int ny, int nz, ValueType* coeffs)
{
    const auto coarse_dp = mat->get_size()[0];
    const auto fine_dp = mat->get_size()[1];
    GKO_ASSERT((nx + 1) * (ny + 1) * (nz + 1) == coarse_dp);
    GKO_ASSERT((2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1) == fine_dp);
    matrix_data<ValueType, IndexType> data{dim<2>(coarse_dp, fine_dp), {}};

#pragma omp for
    for (auto iz = 0; iz <= nz; iz++) {
        for (auto iy = 0; iy <= ny; iy++) {
            for (auto ix = 0; ix <= nx; ix++) {
                auto current_row = grid2index(ix, iy, iz, nx, ny);

                const auto on_border = (ix == 0) | (iy == 0) | (iz == 0) |
                                       (ix == nx) | (iy == ny) | (iz == nz);
                auto fine_gridpoint =
                    grid2index(2 * ix, 2 * iy, 2 * iz, 2 * nx, 2 * ny);
                if (!on_border) {
                    for (auto ofs_z : {-1, 0, 1}) {
                        if (iz + ofs_z > -1 && iz + ofs_z <= nz) {
                            for (auto ofs_y : {-1, 0, 1}) {
                                if (iy + ofs_y > -1 && iy + ofs_y <= ny) {
                                    for (auto ofs_x : {-1, 0, 1}) {
                                        if (ix + ofs_x > -1 &&
                                            ix + ofs_x <= nx) {
                                            auto current_col = grid2index(
                                                ofs_x, ofs_y, ofs_z, 2 * nx,
                                                2 * ny,
                                                fine_gridpoint);  // index of
                                                                  // the point
                                                                  // on the fine
                                                                  // grid
                                            auto tmp = coeffs[ofs_z + 1] *
                                                       coeffs[ofs_y + 1] *
                                                       coeffs[ofs_x + 1];
                                            data.nonzeros.emplace_back(
                                                current_row, current_col, tmp);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    data.nonzeros.emplace_back(current_row, fine_gridpoint, 1);
                }
            }
        }
    }
    data.ensure_row_major_order();
    mat->read(data);
}

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
