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


#include "core/constraints/constrained_system_kernels.hpp"

#include <memory>

#include <ginkgo/core/base/array.hpp>


namespace gko {
namespace kernels {
namespace reference {
namespace cons {


template <typename ValueType, typename IndexType>
void fill_subset(std::shared_ptr<const DefaultExecutor> exec,
                 const Array<IndexType>& subset, ValueType* data, ValueType val)
{
    const auto* idxs = subset.get_const_data();
    for (int i = 0; i < subset.get_num_elems(); ++i) {
        data[idxs[i]] = val;
    }
}


template <typename ValueType, typename IndexType>
void copy_subset(std::shared_ptr<const DefaultExecutor> exec,
                 const Array<IndexType>& subset, const ValueType* src,
                 ValueType* dst)
{
    const auto* idxs = subset.get_const_data();
    for (int i = 0; i < subset.get_num_elems(); ++i) {
        dst[idxs[i]] = src[idxs[i]];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CONS_FILL_SUBSET);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CONS_COPY_SUBSET);


}  // namespace cons
}  // namespace reference
}  // namespace kernels
}  // namespace gko
