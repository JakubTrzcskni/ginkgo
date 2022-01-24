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

#include "core/solver/minres_kernels.hpp"

#include <ginkgo/core/base/executor.hpp>


#include "common/unified/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Minres solver namespace.
 *
 * @ingroup minres
 */
namespace minres {


template <typename ValueType>
void initialize(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Dense<ValueType>* r, matrix::Dense<ValueType>* z,
    matrix::Dense<ValueType>* p, matrix::Dense<ValueType>* p_prev,
    matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* q_prev,
    matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* gamma,
    matrix::Dense<ValueType>* delta, matrix::Dense<ValueType>* cos_prev,
    matrix::Dense<ValueType>* cos, matrix::Dense<ValueType>* sin_prev,
    matrix::Dense<ValueType>* sin, matrix::Dense<ValueType>* eta_next,
    matrix::Dense<ValueType>* eta, Array<stopping_status>* stop_status)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto beta, auto gamma, auto delta,
                      auto cos_prev, auto cos, auto sin_prev, auto sin,
                      auto eta_next, auto eta, auto stop) {
            delta[col] = gamma[col] = cos_prev[col] = sin_prev[col] = sin[col] =
                zero(*delta);
            cos[col] = one(*delta);
            eta_next[col] = eta[col] = beta[col] = sqrt(beta[col]);
            stop[col].reset();
        },
        beta->get_num_stored_elements(), row_vector(beta), row_vector(gamma),
        row_vector(delta), row_vector(cos_prev), row_vector(cos),
        row_vector(sin_prev), row_vector(sin), row_vector(eta_next),
        row_vector(eta), *stop_status);

    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto r, auto z, auto p, auto p_prev,
                      auto q, auto q_prev, auto beta, auto stop) {
            q(row, col) = safe_divide(r(row, col), beta[col]);
            z(row, col) = safe_divide(z(row, col), beta[col]);
            p(row, col) = p_prev(row, col) = q_prev(row, col) =
                zero(p(row, col));
        },
        r->get_size(), r->get_stride(), default_stride(r), default_stride(z),
        default_stride(p), default_stride(p_prev), default_stride(q),
        default_stride(q_prev), row_vector(beta), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* p,
            matrix::Dense<ValueType>* p_prev, matrix::Dense<ValueType>* z,
            const matrix::Dense<ValueType>* z_tilde,
            matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* q_prev,
            const matrix::Dense<ValueType>* q_tilde,
            matrix::Dense<ValueType>* alpha, matrix::Dense<ValueType>* beta,
            matrix::Dense<ValueType>* gamma, matrix::Dense<ValueType>* delta,
            matrix::Dense<ValueType>* cos_prev, matrix::Dense<ValueType>* cos,
            matrix::Dense<ValueType>* sin_prev, matrix::Dense<ValueType>* sin,
            matrix::Dense<ValueType>* eta, matrix::Dense<ValueType>* eta_next,
            matrix::Dense<ValueType>* tau,
            const Array<stopping_status>* stop_status)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto beta, auto stop) {
            if (!stop[col].has_stopped()) {
                beta[col] = sqrt(beta[col]);
            }
        },
        beta->get_num_stored_elements(), row_vector(beta), *stop_status);

    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto q, auto q_prev, auto q_tilde,
                      auto beta, auto stop) {
            if (!stop[col].has_stopped()) {
                q_prev(row, col) = q(row, col);
                q(row, col) = safe_divide(q_tilde(row, col), beta[col]);
            }
        },
        q->get_size(), q->get_stride(), default_stride(q),
        default_stride(q_prev), default_stride(q_tilde), row_vector(beta),
        *stop_status);

    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto alpha, auto beta, auto gamma, auto delta,
                      auto cos_prev, auto cos, auto sin_prev, auto sin,
                      auto eta_next, auto eta, auto tau, auto stop) {
            if (!stop[col].has_stopped()) {
                delta[col] = sin_prev[col] * gamma[col];
                const auto tmp_d = gamma[col];
                const auto tmp_a = alpha[col];
                gamma[col] =
                    cos_prev[col] * cos[col] * tmp_d + sin[col] * tmp_a;
                alpha[col] =
                    -conj(sin[col]) * cos_prev[col] * tmp_d + cos[col] * tmp_a;
            }
        },
        alpha->get_num_stored_elements(), row_vector(alpha), row_vector(beta),
        row_vector(gamma), row_vector(delta), row_vector(cos_prev),
        row_vector(cos), row_vector(sin_prev), row_vector(sin),
        row_vector(eta_next), row_vector(eta), row_vector(tau), *stop_status);

    std::swap(*cos, *cos_prev);
    std::swap(*sin, *sin_prev);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto alpha, auto beta, auto gamma, auto delta,
                      auto cos_prev, auto cos, auto sin_prev, auto sin,
                      auto eta_next, auto eta, auto tau, auto stop) {
            if (!stop[col].has_stopped()) {
                if (alpha[col] == zero(alpha[col])) {
                    cos[col] = zero(cos[col]);
                    sin[col] = one(sin[col]);
                } else {
                    const auto scale = abs(alpha[col]) + abs(beta[col]);
                    const auto hypotenuse =
                        scale *
                        sqrt(abs(alpha[col] / scale) * abs(alpha[col] / scale) +
                             abs(beta[col] / scale) * abs(beta[col] / scale));
                    cos[col] = conj(alpha[col]) / hypotenuse;
                    sin[col] = conj(beta[col]) / hypotenuse;
                }
            }
        },
        alpha->get_num_stored_elements(), row_vector(alpha), row_vector(beta),
        row_vector(gamma), row_vector(delta), row_vector(cos_prev),
        row_vector(cos), row_vector(sin_prev), row_vector(sin),
        row_vector(eta_next), row_vector(eta), row_vector(tau), *stop_status);

    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto alpha, auto beta, auto gamma, auto delta,
                      auto cos_prev, auto cos, auto sin_prev, auto sin,
                      auto eta_next, auto eta, auto tau, auto stop) {
            if (!stop[col].has_stopped()) {
                tau[col] = abs(sin[col]) * tau[col];
                eta[col] = eta_next[col];
                eta_next[col] = -conj(sin[col]) * eta[col];
                alpha[col] = cos[col] * alpha[col] + sin[col] * beta[col];
            }
        },
        alpha->get_num_stored_elements(), row_vector(alpha), row_vector(beta),
        row_vector(gamma), row_vector(delta), row_vector(cos_prev),
        row_vector(cos), row_vector(sin_prev), row_vector(sin),
        row_vector(eta_next), row_vector(eta), row_vector(tau), *stop_status);

    std::swap(*p, *p_prev);
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto p, auto p_prev, auto z,
                      auto z_tilde, auto alpha, auto beta, auto gamma,
                      auto delta, auto cos, auto eta, auto stop) {
            if (!stop[col].has_stopped()) {
                p(row, col) =
                    safe_divide(z(row, col) - gamma[col] * p_prev(row, col) -
                                    delta[col] * p(row, col),
                                alpha[col]);
                x(row, col) = x(row, col) + cos[col] * eta[col] * p(row, col);
                z(row, col) = safe_divide(z_tilde(row, col), beta[col]);
            }
        },
        x->get_size(), p->get_stride(), x, default_stride(p),
        default_stride(p_prev), default_stride(z), default_stride(z_tilde),
        row_vector(alpha), row_vector(beta), row_vector(gamma),
        row_vector(delta), row_vector(cos), row_vector(eta), *stop_status);

    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto beta, auto gamma, auto stop) {
            if (!stop[col].has_stopped()) {
                gamma[col] = beta[col];
            }
        },
        beta->get_num_stored_elements(), row_vector(beta), row_vector(gamma),
        *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_STEP_1_KERNEL);


}  // namespace minres
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
