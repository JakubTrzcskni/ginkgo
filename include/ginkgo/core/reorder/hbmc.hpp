// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_REORDER_HBMC_HPP_
#define GKO_PUBLIC_CORE_REORDER_HBMC_HPP_

#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/preconditioner/gauss_seidel.hpp>
#include <ginkgo/core/reorder/reordering_base.hpp>

namespace gko {
namespace reorder {

template <typename ValueType = default_precision, typename IndexType = int32>
class Hbmc : public EnablePolymorphicObject<Hbmc<ValueType, IndexType>,
                                            ReorderingBase<IndexType>>,
             public EnablePolymorphicAssignment<Hbmc<ValueType, IndexType>> {
    friend class EnablePolymorphicObject<Hbmc, ReorderingBase<IndexType>>;

public:
    using PermutationMatrix = matrix::Permutation<IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Gets the permutation (permutation matrix, output of the algorithm) of the
     * linear operator.
     *
     * @return the permutation (permutation matrix)
     */
    std::shared_ptr<const PermutationMatrix> get_permutation() const
    {
        return permutation_;
    }

    /**
     * Gets the inverse permutation (permutation matrix, output of the
     * algorithm) of the linear operator.
     *
     * @return the inverse permutation (permutation matrix)
     */
    std::shared_ptr<const PermutationMatrix> get_inverse_permutation() const
    {
        return inv_permutation_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * If this parameter is set then an inverse permutation matrix is also
         * constructed along with the normal permutation matrix.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(construct_inverse_permutation, false);

        size_t GKO_FACTORY_PARAMETER_SCALAR(base_block_size, 4u);

        size_t GKO_FACTORY_PARAMETER_SCALAR(lvl_2_block_size, 32u);
    };
    GKO_ENABLE_REORDERING_BASE_FACTORY(Hbmc, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit Hbmc(std::shared_ptr<const Executor> exec);

    explicit Hbmc(const Factory* factory, const ReorderingBaseArgs& args);

private:
    std::shared_ptr<PermutationMatrix> permutation_;
    std::shared_ptr<PermutationMatrix> inv_permutation_;
};

}  // namespace reorder

namespace experimental {
namespace reorder {


template <typename IndexType = int32>
class Hbmc : public EnablePolymorphicObject<Hbmc<IndexType>, LinOpFactory>,
             public EnablePolymorphicAssignment<Hbmc<IndexType>> {
public:
    struct parameters_type;
    friend class EnablePolymorphicObject<Hbmc<IndexType>, LinOpFactory>;
    friend class enable_parameters_type<parameters_type, Hbmc<IndexType>>;

    using index_type = IndexType;
    using permutation_type = matrix::Permutation<index_type>;

    struct parameters_type
        : public enable_parameters_type<parameters_type, Hbmc<IndexType>> {
        size_t GKO_FACTORY_PARAMETER_SCALAR(base_block_size, 4u);

        size_t GKO_FACTORY_PARAMETER_SCALAR(lvl_2_block_size, 32u);

        bool GKO_FACTORY_PARAMETER_SCALAR(padding, false);
    };
    /**
     * Returns the parameters used to construct the factory.
     *
     * @return the parameters used to construct the factory.
     */
    const parameters_type& get_parameters() { return parameters_; }

    /**
     * @copydoc LinOpFactory::generate
     * @note This function overrides the default LinOpFactory::generate to
     *       return a Permutation instead of a generic LinOp, which would
     *       need to be cast to Permutation again to access its indices.
     *       It is only necessary because smart pointers aren't covariant.
     */
    std::unique_ptr<permutation_type> generate(
        std::shared_ptr<const LinOp> system_matrix) const;

    /** Creates a new parameter_type to set up the factory. */
    static parameters_type build() { return {}; }

protected:
    explicit Hbmc(std::shared_ptr<const Executor> exec,
                  const parameters_type& params = {});

    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> system_matrix) const override;

    parameters_type parameters_;
};


}  // namespace reorder
}  // namespace experimental
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_HBMC_HPP_