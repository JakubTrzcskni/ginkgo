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


#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_IDENTITY_HPP
#define GKO_REFERENCE_PRECONDITIONER_BATCH_IDENTITY_HPP


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace reference {


/**
 * Identity preconditioner for batch solvers.(Effectively unpreconditioned
 * solver)
 */
template <typename ValueType>
class BatchIdentity final {
public:
    /**
     * The size of the work vector required in case of static allocation.
     */
    static constexpr int work_size = 0;

    /**
     * The size of the work vector required in case of dynamic allocation.
     *
     * For the Identity preconditioner, this is unnecessary, but this function
     * is part of a 'batch preconditioner interface' because other
     * preconditioners may need it.
     */
    static int dynamic_work_size(int, int) { return 0; }

    /**
     * Sets the input and generates identity preconditioner(In reality, it
     * generates nothing (work size array is 0) as application of identity
     * preconditioner is trivial)
     *
     * @param mat  Matrix for which to build an Identity preconditioner.
     * @param work  A 'work-vector', which is unnecessary here. It is not
     * actually required, nor does it store anything.
     */
    BatchIdentity(const gko::batch_csr::BatchEntry<const ValueType> &mat,
                  ValueType *const work)
        : matrix_{mat}, work_{work}
    {}

    void apply(const gko::batch_dense::BatchEntry<const ValueType> &r,
               const gko::batch_dense::BatchEntry<ValueType> &z) const
    {
        for (int i = 0; i < matrix_.num_rows; i++) {
            for (int j = 0; j < r.num_rhs; j++)
                z.values[i * z.stride + j] = r.values[i * r.stride + j];
        }
    }

private:
    ValueType *const work_;
    const gko::batch_csr::BatchEntry<const ValueType> &matrix_;
};


}  // namespace reference
}  // namespace kernels
}  // namespace gko

#endif