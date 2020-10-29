/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/stop/residual_norm_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Residual norm stopping criterion namespace.
 * @ref resnorm
 * @ingroup resnorm
 */
namespace residual_norm {


constexpr int default_block_size = 512;


#include "common/stop/residual_norm_kernel.hpp.inc"


template <typename ValueType>
void residual_norm(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *tau,
                   const matrix::Dense<ValueType> *orig_tau,
                   ValueType rel_residual_goal, uint8 stoppingId,
                   bool setFinalized, Array<stopping_status> *stop_status,
                   Array<bool> *device_storage, bool *all_converged,
                   bool *one_changed)
{
    static_assert(is_complex_s<ValueType>::value == false,
                  "ValueType must not be complex in this function!");
    init_kernel<<<1, 1>>>(as_cuda_type(device_storage->get_data()));

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(tau->get_size()[1], block_size.x), 1, 1);

    residual_norm_kernel<<<grid_size, block_size>>>(
        tau->get_size()[1], rel_residual_goal,
        as_cuda_type(tau->get_const_values()),
        as_cuda_type(orig_tau->get_const_values()), stoppingId, setFinalized,
        as_cuda_type(stop_status->get_data()),
        as_cuda_type(device_storage->get_data()));

    /* Represents all_converged, one_changed */
    *all_converged = exec->copy_val_to_host(device_storage->get_const_data());
    *one_changed = exec->copy_val_to_host(device_storage->get_const_data() + 1);
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_RESIDUAL_NORM_KERNEL);


}  // namespace residual_norm
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
