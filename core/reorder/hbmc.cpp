// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/hbmc.hpp>

#include <memory>


namespace gko {
namespace reorder {

// template <typename ValueType, typename IndexType>
// void hbmc_reorder(const matrix::Csr<ValueType, IndexType>* mtx,
//                   IndexType* permutation, IndexType* inv_permutation)
// {}

template <typename ValueType, typename IndexType>
Hbmc<ValueType, IndexType>::Hbmc(std::shared_ptr<const Executor> exec)
    : EnablePolymorphicObject<Hbmc, ReorderingBase<IndexType>>(std::move(exec))
{}


template <typename ValueType, typename IndexType>
Hbmc<ValueType, IndexType>::Hbmc(const Factory* factory,
                                 const ReorderingBaseArgs& args)
    : EnablePolymorphicObject<Hbmc, ReorderingBase<IndexType>>(
          factory->get_executor()),
      parameters_{factory->get_parameters()}
{
    auto base_block_size = parameters_.base_block_size;
    auto lvl_2_block_size = parameters_.lvl_2_block_size;

    auto hbmc_gs = preconditioner::GaussSeidel<ValueType, IndexType>::build()
                       .with_base_block_size(base_block_size)
                       .with_lvl_2_block_size(lvl_2_block_size)
                       .on(this->get_executor()->get_master())
                       ->generate(args.system_matrix);
    auto permutation_array = hbmc_gs->get_permutation_idxs();
    permutation_ =
        PermutationMatrix::create(this->get_executor(), permutation_array);

    inv_permutation_ = nullptr;
    this->set_permutation_array(permutation_array);
}

#define GKO_DECLARE_HBMC(ValueType, IndexType) class Hbmc<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_HBMC);

}  // namespace reorder

namespace experimental {
namespace reorder {

template <typename IndexType>
Hbmc<IndexType>::Hbmc(std::shared_ptr<const Executor> exec,
                      const parameters_type& params)
    : EnablePolymorphicObject<Hbmc, LinOpFactory>(std::move(exec)),
      parameters_{params}
{}


template <typename IndexType>
std::unique_ptr<matrix::Permutation<IndexType>> Hbmc<IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    auto product =
        std::unique_ptr<permutation_type>(static_cast<permutation_type*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename IndexType>
std::unique_ptr<LinOp> Hbmc<IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    using ValueType = float;
    using Mtx = matrix::Csr<ValueType, IndexType>;
    auto base_block_size = parameters_.base_block_size;
    auto lvl_2_block_size = parameters_.lvl_2_block_size;
    auto padding = parameters_.padding;

    auto hbmc_gs = preconditioner::GaussSeidel<ValueType, IndexType>::build()
                       .with_use_HBMC(true)
                       .with_base_block_size(base_block_size)
                       .with_lvl_2_block_size(lvl_2_block_size)
                       .with_use_padding(padding)
                       .on(this->get_executor()->get_master())
                       ->generate(system_matrix);
    auto permutation_array = hbmc_gs->get_permutation_idxs();

    return permutation_type::create(this->get_executor(),
                                    std::move(permutation_array));
}


#undef GKO_DECLARE_HBMC
#define GKO_DECLARE_HBMC(IndexType) class Hbmc<IndexType>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_HBMC);
}  // namespace reorder
}  // namespace experimental
}  // namespace gko
