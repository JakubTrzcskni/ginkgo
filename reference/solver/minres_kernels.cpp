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


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
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
    for (size_type j = 0; j < r->get_size()[1]; ++j) {
        delta->at(j) = gamma->at(j) = cos_prev->at(j) = sin_prev->at(j) =
            sin->at(j) = zero<ValueType>();
        cos->at(j) = one<ValueType>();
        eta_next->at(j) = eta->at(j) = beta->at(j) = sqrt(beta->at(j));
        stop_status->get_data()[j].reset();
    }
    for (size_type i = 0; i < r->get_size()[0]; ++i) {
        for (size_type j = 0; j < r->get_size()[1]; ++j) {
            q->at(i, j) = safe_divide(r->at(i, j), beta->at(j));
            z->at(i, j) = safe_divide(z->at(i, j), beta->at(j));
            p->at(i, j) = p_prev->at(i, j) = q_prev->at(i, j) =
                zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_INITIALIZE_KERNEL);


template <typename ValueType>
void compute_givens_rotation(const matrix::Dense<ValueType>* alpha,
                             const matrix::Dense<ValueType>* beta,
                             matrix::Dense<ValueType>* cos,
                             matrix::Dense<ValueType>* sin,
                             const Array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < beta->get_size()[1]; ++j) {
        if (!stop_status->get_const_data()[j].has_stopped()) {
            if (alpha->at(j) == zero<ValueType>()) {
                cos->at(j) = zero<ValueType>();
                sin->at(j) = one<ValueType>();
            } else {
                const auto scale = abs(alpha->at(j)) + abs(beta->at(j));
                const auto hypotenuse =
                    scale *
                    sqrt(abs(alpha->at(j) / scale) * abs(alpha->at(j) / scale) +
                         abs(beta->at(j) / scale) * abs(beta->at(j) / scale));
                cos->at(j) = conj(alpha->at(j)) / hypotenuse;
                sin->at(j) = conj(beta->at(j)) / hypotenuse;
            }
        }
    }
}


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
    /**
     * beta = sqrt(beta)
     * q_-1 = q
     * q = q_tilde / beta
     */
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        beta->at(j) = sqrt(beta->at(j));
    }
    for (size_type i = 0; i < x->get_size()[0]; ++i) {
        for (size_type j = 0; j < x->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            q_prev->at(i, j) = q->at(i, j);
            q->at(i, j) = safe_divide(q_tilde->at(i, j), beta->at(j));
        }
    }

    /*
     * apply two previous givens rot to new column
     * delta = s_-1 * gamma  // 0 if iter = 0, 1
     * tmp_g = gamma
     * tmp_a = alpha
     * gamma = c * c_-1 * tmp_g + s * tmp_a  // 0 if iter = 0
     * alpha = -conj(s) * c_-1 * tmp_g + c * tmp_a
     */
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        delta->at(j) = sin_prev->at(j) * gamma->at(j);
        auto tmp_d = gamma->at(j);
        auto tmp_a = alpha->at(j);
        gamma->at(j) =
            cos_prev->at(j) * cos->at(j) * tmp_d + sin->at(j) * tmp_a;
        alpha->at(j) =
            -conj(sin->at(j)) * cos_prev->at(j) * tmp_d + cos->at(j) * tmp_a;
    }

    /*
     * compute new givens rot
     * s_-1 = s
     * c_-1 = c
     * c, s = givens_rot(alpha, beta)
     *
     */
    std::swap(*cos, *cos_prev);
    std::swap(*sin, *sin_prev);
    compute_givens_rotation(alpha, beta, cos, sin, stop_status);

    /*
     * apply new givens rot to T and eta
     * eta = eta_+1
     * eta_+1 = -conj(s) * eta
     * alpha = c * alpha + s * beta
     * beta = -conj(s) * alpha + c * beta = 0
     */
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        tau->at(j) = abs(sin->at(j)) * tau->at(j);
        eta->at(j) = eta_next->at(j);
        eta_next->at(j) = -conj(sin->at(j)) * eta->at(j);
        alpha->at(j) = cos->at(j) * alpha->at(j) + sin->at(j) * beta->at(j);
    }

    /*
     * update search direction and solution
     * swap(p, p_-1)
     * p = (z - gamma * p_-1 - delta * p) / alpha
     * x = x + c * eta * p
     *
     * z = z_tilde / beta  // lanzcos continuation
     *
     * prepare next iteration with
     * gamma = beta
     * beta = -beta
     */
    std::swap(*p, *p_prev);
    for (size_type i = 0; i < x->get_size()[0]; ++i) {
        for (size_type j = 0; j < x->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            p->at(i, j) =
                safe_divide(z->at(i, j) - gamma->at(j) * p_prev->at(i, j) -
                                delta->at(j) * p->at(i, j),
                            alpha->at(j));
            x->at(i, j) = x->at(i, j) + cos->at(j) * eta->at(j) * p->at(i, j);
            z->at(i, j) = safe_divide(z_tilde->at(i, j), beta->at(j));
        }
    }
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        gamma->at(j) = beta->at(j);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_STEP_1_KERNEL);


}  // namespace minres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
