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

#include <ginkgo/core/solver/minres.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>


#include "core/solver/minres_kernels.hpp"


namespace gko {
namespace solver {
namespace minres {
namespace {


GKO_REGISTER_OPERATION(initialize, minres::initialize);
GKO_REGISTER_OPERATION(step_1, minres::step_1);


}  // anonymous namespace
}  // namespace minres


template <typename ValueType>
std::unique_ptr<LinOp> Minres<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->stop_criterion_factory_)
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Minres<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->stop_criterion_factory_)
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Minres<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
void Minres<ValueType>::apply_dense_impl(
    const matrix::Dense<ValueType>* dense_b,
    matrix::Dense<ValueType>* dense_x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto r = Vector::create_with_config_of(dense_b);
    auto z = Vector::create_with_config_of(dense_b);
    auto p = Vector::create_with_config_of(dense_b);
    auto q = Vector::create_with_config_of(dense_b);

    auto z_tilde = Vector::create_with_config_of(dense_b);
    auto q_tilde = Vector::create_with_config_of(dense_b);

    auto p_prev = Vector::create_with_config_of(p.get());
    auto q_prev = Vector::create_with_config_of(q.get());

    auto alpha = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto beta = Vector::create_with_config_of(alpha.get());
    auto gamma = Vector::create_with_config_of(alpha.get());
    auto delta = Vector::create_with_config_of(alpha.get());
    auto eta_next = Vector::create_with_config_of(alpha.get());
    auto eta = Vector::create_with_config_of(alpha.get());
    auto tau = Vector::create_with_config_of(
        alpha.get());  // using ||z||, could also be beta or ||r||?

    auto cos_prev = Vector::create_with_config_of(alpha.get());
    auto cos = Vector::create_with_config_of(alpha.get());
    auto sin_prev = Vector::create_with_config_of(alpha.get());
    auto sin = Vector::create_with_config_of(alpha.get());

    bool one_changed{};
    Array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // r = dense_b
    // eta = eta_next = norm(dense_b - A * dense_x)
    // z = p = q = 0
    r = clone(dense_b);
    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_,
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        r.get());

    get_preconditioner()->apply(r.get(), z.get());
    r->compute_conj_dot(z.get(), beta.get());
    z->compute_norm2(tau.get());

    exec->run(minres::make_initialize(
        r.get(), z.get(), p.get(), p_prev.get(), q.get(), q_prev.get(),
        beta.get(), gamma.get(), delta.get(), cos_prev.get(), cos.get(),
        sin_prev.get(), sin.get(), eta_next.get(), eta.get(), &stop_status));

    int iter = -1;
    /* Memory movement summary:
     * 18n * values + matrix/preconditioner storage
     * 1x SpMV:           2n * values + storage
     * 1x Preconditioner: 2n * values + storage
     * 2x dot             4n
     * 1x step 1 (axpy)   3n
     * 1x step 2 (axpys)  6n
     * 1x norm2 residual   n
     */
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, r.get(), dense_x, tau.get(), nullptr);
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .residual_norm(tau.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        /**
         * Lanzcos (partial) update
         *
         * q_tilde = A * z - beta * q_prev
         * alpha = dot(q_tilde, z)
         * q_tilde = q_tilde - alpha * q
         * z_tilde = M * q_tilde
         * beta = dot(q_tilde, z_tilde);
         *
         * p and z get updated in step_1
         */
        swap(q_tilde, q_prev);
        q_tilde->scale(beta.get());
        system_matrix_->apply(one_op.get(), z.get(), neg_one_op.get(),
                              q_tilde.get());
        q_tilde->compute_conj_dot(z.get(), alpha.get());
        q_tilde->sub_scaled(alpha.get(), q.get());
        get_preconditioner()->apply(q_tilde.get(), z_tilde.get());
        q_tilde->compute_conj_dot(z_tilde.get(), beta.get());

        /**
         * step 2:
         * finish lanzcos pt1
         * beta = sqrt(beta)
         * q_-1 = q
         * q = q_tilde / beta
         *
         * apply two previous givens rot to new column
         * delta = s_-1 * gamma  // 0 if iter = 0, 1
         * tmp_d = gamma
         * tmp_a = alpha
         * gamma = c_-1 * c * tmp_d + s * tmp_a  // 0 if iter = 0
         * alpha = -conj(s) * c_-1 * tmp_d + c * tmp_a
         *
         * compute new givens rot
         * s_-1 = s
         * c_-1 = c
         * c, s = givens_rot(alpha, beta)
         *
         * apply new givens rot to T and eta
         * tau = abs(s) * tau
         * eta = eta_+1
         * eta_+1 = -conj(s) * eta
         * alpha = c * alpha + s * beta
         *
         * update search direction and solution
         * swap(p, p_-1)
         * p = (z - gamma * p_-1 - delta * p) / alpha
         * x = x + c * eta * p
         *
         * finish lanzcos pt2
         * z = z_tilde / beta  // lanzcos continuation
         */
        exec->run(minres::make_step_1(
            dense_x, p.get(), p_prev.get(), z.get(), z_tilde.get(), q.get(),
            q_prev.get(), q_tilde.get(), alpha.get(), beta.get(), gamma.get(),
            delta.get(), cos_prev.get(), cos.get(), sin_prev.get(), sin.get(),
            eta.get(), eta_next.get(), tau.get(), &stop_status));
    }
}


template <typename ValueType>
void Minres<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                   const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_MINRES(_type) class Minres<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES);


}  // namespace solver
}  // namespace gko
