#ifndef MULTIGRID_TEST
#define MULTIGRID_TEST
#include <ginkgo/ginkgo.hpp>
#include "geometric-multigrid.hpp"
using namespace gko::multigrid;

template <typename ValueType, typename IndexType>
void test_matrix_generation(std::shared_ptr<const gko::Executor> exec,
                            problem_geometry& geometry, ValueType value_help,
                            IndexType index_help);
template <typename ValueType, typename IndexType>
void test_restriction(std::shared_ptr<const gko::Executor> exec,
                      problem_geometry& geometry, ValueType value_help,
                      IndexType index_help);

#endif