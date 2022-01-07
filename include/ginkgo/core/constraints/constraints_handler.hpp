/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_CORE_CONSTRAINTS_CONSTRAINTS_HANDLER_HPP_
#define GKO_PUBLIC_CORE_CONSTRAINTS_CONSTRAINTS_HANDLER_HPP_


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/matrix/dense.hpp>

namespace gko {
namespace constraints {
namespace detail {


/**
 * Creates a dense vector with fixed values on specific indices and zero
 * elsewhere.
 *
 * @param size  Total size of the vector.
 * @param idxs  The indices that should be set.
 * @param values  The values that should be set.
 */
template <typename ValueType, typename IndexType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
zero_guess_with_constrained_values(std::shared_ptr<const Executor> exec,
                                   dim<2> size, const IndexSet<IndexType>& idxs,
                                   const matrix::Dense<ValueType>* values);


}  // namespace detail


/**
 * Interface for applying constraints to a linear system.
 *
 * This interface provides several methods that are necessary to construct the
 * individual parts of a linear system with constraints, namely:
 * - incorporating the constraints into the operator
 * - deriving a suitable right-hand-side
 * - deriving a suitable initial guess
 * - if necessary, update the solution.
 * Depending on the actual implementation, some of these methods might be
 * no-ops.
 *
 * A specialized implementation of constraints handling can be achieved by
 * deriving a class from this interface and passing it to the
 * ConstraintsHandler.
 *
 * @tparam ValueType The ValueType of the underlying operator and vectors
 * @tparam IndexType The IndexType of the underlying operator and vectors
 */
template <typename ValueType, typename IndexType>
class ApplyConstraintsStrategy
    : public EnableCreateMethod<
          ApplyConstraintsStrategy<ValueType, ValueType>> {
public:
    /**
     * Incorporates the constraints into the operator.
     *
     * @note This might (but not necessarily) change the operator directly.
     *
     * @param idxs  The indices where the constraints are applied.
     * @param op  The original operator.
     * @return  An operator with constraints added.
     */
    virtual std::shared_ptr<LinOp> construct_operator(
        const IndexSet<IndexType>& idxs, std::shared_ptr<LinOp> op) = 0;

    /**
     * Creates a new right-hand-side for the constrained system.
     *
     * @param idxs  The indices where the constraints are applied.
     * @param op  The original (unconstrained) operator.
     * @param init_guess  The original initial guess of the system
     * @param rhs  The original right-hand-side.
     * @return  The right-hand-side for the constrained system.
     */
    virtual std::unique_ptr<LinOp> construct_right_hand_side(
        const IndexSet<IndexType>& idxs, const LinOp* op,
        const matrix::Dense<ValueType>* init_guess,
        const matrix::Dense<ValueType>* rhs) = 0;

    /**
     * Creates a new initial guess for the constrained system.
     *
     * @param idxs  The indices where the constraints are applied.
     * @param op  The original (unconstrained) operator.
     * @param init_guess  The original initial guess of the system
     * @param constrained_values  The values of the constrained indices.
     * @return  A new initial guess for the constrained system.
     */
    virtual std::unique_ptr<LinOp> construct_initial_guess(
        const IndexSet<IndexType>& idxs, const LinOp* op,
        const matrix::Dense<ValueType>* init_guess,
        const matrix::Dense<ValueType>* constrained_values) = 0;

    /**
     * If necessary, updates the solution to the constrained system to make it
     * the solution of the original system.
     *
     * @param idxs  The indices where the constraints are applied.
     * @param constrained_values  The values of the constrained indices.
     * @param orig_init_guess The original (unconstrained) initial guess of the
     * system
     * @param solution The solution to the constrained system.
     */
    virtual void correct_solution(
        const IndexSet<IndexType>& idxs,
        const matrix::Dense<ValueType>* constrained_values,
        const matrix::Dense<ValueType>* orig_init_guess,
        matrix::Dense<ValueType>* solution) = 0;
};


/**
 * Applies constraints to a linear system, by zeroing-out rows.
 *
 * The rows of a matrix that correspond to constrained values are set to zero,
 * except the diagonal entry, which is set to 1. This directly changes the
 * values of the matrix, and the operator's symmetry is not maintained. However,
 * the operator is still symmetric (or self-adjoint) on a subspace, where the
 * constrained indices of vectors are set to zero, so that the constrained
 * operator can still be used in a CG method for example. Additionally, a new
 * right-hand-side in that subspace is created as `new_b = b - cons_A * x_0` and
 * the new initial guess is set to 0 for constrained indices. Lastly, the
 * constrained values are added to the solution of the system `cons_A * z =
 * new_b`.
 *
 * @note Current restrictions:
 * - can only be used with a single right-hand-side
 * - can only be used with `stride=1` vectors
 *
 * @tparam ValueType The ValueType of the underlying operator and vectors
 * @tparam IndexType The IndexType of the underlying operator and vectors
 */
template <typename ValueType, typename IndexType>
class ZeroRowsStrategy : public ApplyConstraintsStrategy<ValueType, IndexType> {
    using Dense = matrix::Dense<ValueType>;

public:
    std::shared_ptr<LinOp> construct_operator(
        const IndexSet<IndexType>& idxs, std::shared_ptr<LinOp> op) override;

    std::unique_ptr<LinOp> construct_right_hand_side(
        const IndexSet<IndexType>& idxs, const LinOp* op,
        const matrix::Dense<ValueType>* init_guess,
        const matrix::Dense<ValueType>* rhs) override;

    std::unique_ptr<LinOp> construct_initial_guess(
        const IndexSet<IndexType>& idxs, const LinOp* op,
        const matrix::Dense<ValueType>* init_guess,
        const matrix::Dense<ValueType>* constrained_values) override;

    void correct_solution(const IndexSet<IndexType>& idxs,
                          const matrix::Dense<ValueType>* constrained_values,
                          const matrix::Dense<ValueType>* orig_init_guess,
                          matrix::Dense<ValueType>* solution) override;

private:
    std::unique_ptr<Dense> one;
    std::unique_ptr<Dense> neg_one;
};


/**
 * A class that handles a linear system with constraints.
 *
 * The ConstraintsHandler can incorporate constraints of the form `x_i = g_i`
 * for some indices `i` into a given linear system `Ax = b`. This can be used
 * for example, to incorporate Dirichlet conditions into a linear system.
 *
 * To solve a given linear system with matrix `A` and right-hand-side `b` and
 * constraints on the indices `idxs` with values `values`, the
 * ConstraintsHandler can be used in the following way:
 * ```c++
 * ConstraintsHandler handler(idxs, A, values, b,  x,
 *                            std::make_unique<ZeroRowStrategy>());
 * auto cons_A = handler.get_operator(); // cons_A is a shared_ptr
 * auto cons_b = handler.get_right_hand_side(); // cons_b is a bare ptr
 * auto cons_x = handler.get_get_initial_guess(); // cons_x is a bare_ptr
 *
 * auto solver = factory->generate(lend(cons_A));
 * solver->apply(cons_b, cons_x);
 * handler.correct_solution(cons_x);
 * ```
 * Additionally, the constrained values, right-hand-side, and initial guess of
 * the constrained system can be set/updated individually. This can be achieved
 * using the `with_*` functions. For example, to update the constrained values
 * of a linear system that is solved iteratively during multiple time steps one
 * can use the following:
 * ```c++
 * ConstraintsHandler handler(...);
 * auto cons_A = handler.get_operator();
 * for(int time_step=0; time_step < max; time_step++){
 *   auto cons_values = gko::share(create_constrained_values(...));
 *   // invalidates previous results from handler.get_right_hand_side()
 *   // and handler.get_initial_guess()
 *   handler.with_constrained_values(cons_values);
 *
 *   auto cons_b = handler.get_right_hand_side();
 *   auto cons_x = handler.get_get_initial_guess();
 *   solver->apply(cons_b, cons_x);
 *   handler.correct_solution(cons_x);
 * }
 * ```
 *
 * @tparam ValueType The ValueType of the underlying operator
 * @tparam IndexType The IndexType of the underlying operator
 */
template <typename ValueType, typename IndexType>
class ConstraintsHandler {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Dense = matrix::Dense<value_type>;

    /**
     * Initializes the constrained system.
     *
     * Applies the constraints to the system operator and constructs the new
     * initial guess and right hand side using the specified strategy. If no
     * initial guess is supplied, the guess is set to zero.
     *
     * @param idxs  the indices of the constrained degrees of freedom
     * @param system_operator  the original system operator
     * @param values  the values of the constrained degrees of freedom
     * @param right_hand_side  the original right-hand-side of the system
     * @param initial_guess  the initial guess for the original system, has to
     * contain the constrained values. If it is null, then zero will be used as
     * initial guess
     * @param strategy  the implementation strategy of the constraints
     */
    ConstraintsHandler(
        IndexSet<IndexType> idxs, std::shared_ptr<LinOp> system_operator,
        std::shared_ptr<const Dense> values,
        std::shared_ptr<const Dense> right_hand_side,
        std::shared_ptr<const Dense> initial_guess = nullptr,
        std::unique_ptr<ApplyConstraintsStrategy<ValueType, IndexType>>
            strategy =
                std::make_unique<ZeroRowsStrategy<ValueType, IndexType>>());

    /**
     * Initializes the constrained system.
     *
     * Applies the constraints to the system operator using the specified
     * strategy. The constrained values, right-hand-side, and initial guess have
     * to be set later on via the with_* functions.
     *
     * @param idxs  the indices of the constrained degrees of freedom
     * @param system_operator  the original system operator
     * @param strategy  the implementation strategy of the constraints
     */
    ConstraintsHandler(
        IndexSet<IndexType> idxs, std::shared_ptr<LinOp> system_operator,
        std::unique_ptr<ApplyConstraintsStrategy<ValueType, IndexType>>
            strategy =
                std::make_unique<ZeroRowsStrategy<ValueType, IndexType>>());

    /**
     * Sets new contrained values, the corresponding indices are not changed.
     *
     * @note Invalidates previous pointers from get_right_hand_side and
     * get_initial_guess
     *
     * @return *this
     */
    ConstraintsHandler& with_constrained_values(
        std::shared_ptr<const Dense> values);

    /**
     * Set a new right hand side for the linear system.
     *
     * @note Invalidates previous pointers from get_right_hand_side
     *
     * @return *this
     */
    ConstraintsHandler& with_right_hand_side(
        std::shared_ptr<const Dense> right_hand_side);

    /**
     * Set a new initial guess for the linear system. The guess must contain
     * the constrained values.
     *
     * @note Invalidates previous pointers from get_right_hand_side and
     * get_initial_guess
     *
     * @return *this
     */
    ConstraintsHandler& with_initial_guess(
        std::shared_ptr<const Dense> initial_guess);

    /**
     * Read access to the constrained operator
     */
    std::shared_ptr<const LinOp> get_operator() const;

    /**
     * Read access to the right hand side of the constrained system.
     *
     * First call after with_right_hand_side, with_initial_guess, or
     * with_constrained_values constructs the constrained right-hand-side and
     * initial guess if necessary. Without further with_* calls, this function
     * does not recompute the right-hand-side.
     */
    const LinOp* get_right_hand_side();

    /**
     * Read/write access to the initial guess for the constrained system
     *
     * Without providing an initial guess either to the constructor or
     * with_initial_guess, zero will be assumed for the initial guess of the
     * original system.
     *
     * @note Reconstructs the initial guess at every call.
     */
    LinOp* get_initial_guess();

    /**
     * Forces the reconstruction of the constrained system.
     *
     * Afterwards, the modified system can be obtained from get_operator,
     * get_right_hand_side, and get_initial_guess. If no initial guess was
     * provided, the guess will be set to zero.
     */
    void reconstruct_system();

    /**
     * Obtains the solution to the original constrained system from the solution
     * of the modified system
     */
    void correct_solution(Dense* solution);

    LinOp* get_orig_operator() { return lend(orig_operator_); }

    const IndexSet<IndexType>* get_constrained_indices() const
    {
        return &idxs_;
    }

    const Dense* get_constrained_values() const { return lend(values_); }

    const Dense* get_orig_right_hand_side() const { return lend(orig_rhs_); }

    const Dense* get_orig_initial_guess() const
    {
        return lend(orig_init_guess_);
    }

private:
    /**
     * If available, returns the initial guess provided by the user. Otherwise,
     * it will return an initial guess filled with zero.
     */
    std::shared_ptr<const Dense> used_init_guess()
    {
        return orig_init_guess_ ? orig_init_guess_ : zero_init_guess_;
    }

    void reconstruct_system_impl(bool force);

    IndexSet<IndexType> idxs_;

    std::shared_ptr<LinOp> orig_operator_;
    std::shared_ptr<LinOp> cons_operator_;

    std::unique_ptr<ApplyConstraintsStrategy<ValueType, IndexType>> strategy_;

    std::shared_ptr<const Dense> values_;
    std::shared_ptr<const Dense> orig_rhs_;
    std::unique_ptr<Dense> cons_rhs_;
    std::shared_ptr<const Dense> orig_init_guess_;
    std::unique_ptr<Dense> cons_init_guess_;
    std::shared_ptr<const Dense> zero_init_guess_;
};


}  // namespace constraints
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONSTRAINTS_CONSTRAINTS_HANDLER_HPP_
