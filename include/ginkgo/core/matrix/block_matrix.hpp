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

#ifndef GINKGO_BLOCK_MATRIX_HPP
#define GINKGO_BLOCK_MATRIX_HPP


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>


namespace gko {
namespace matrix {


class BlockMatrix : public EnableLinOp<BlockMatrix>,
                    public EnableCreateMethod<BlockMatrix> {
    friend class EnableCreateMethod<BlockMatrix>;
    friend class EnablePolymorphicObject<BlockMatrix, LinOp>;

public:
    static std::unique_ptr<BlockMatrix> create(
        std::shared_ptr<const Executor> exec, const dim<2> size,
        const std::vector<std::vector<std::shared_ptr<LinOp>>>& blocks)
    {
        return std::unique_ptr<BlockMatrix>(
            new BlockMatrix(exec, size, blocks));
    }


    const std::vector<std::vector<std::shared_ptr<LinOp>>>& get_block()
    {
        return blocks_;
    }

    dim<2> get_block_size() const { return block_size_; }

    const std::vector<size_type>& get_size_per_block() const
    {
        return size_per_block_;
    }

protected:
    explicit BlockMatrix(std::shared_ptr<const Executor> exec,
                         const dim<2>& size = {})
        : EnableLinOp<BlockMatrix>(exec, size)
    {}

    BlockMatrix(std::shared_ptr<const Executor> exec, const dim<2> size,
                const std::vector<std::vector<std::shared_ptr<LinOp>>>& blocks)
        : EnableLinOp<BlockMatrix>(exec, size),
          block_size_(blocks.size(), begin(blocks)->size()),
          size_per_block_(blocks.size()),
          blocks_(blocks.size())
    {
        for (size_t row = 0; row < blocks.size(); ++row) {
            blocks_[row] = std::vector(begin(blocks)[row]);
            if (!blocks[row].empty()) {
                size_per_block_[row] = blocks[row][0]->get_size()[0];
            } else {
                size_per_block_[row] = 0;
            }
            GKO_ASSERT_EQ(block_size_[1], blocks_[row].size());
        }
    }

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    dim<2> block_size_;
    std::vector<size_type> size_per_block_;
    std::vector<std::vector<std::shared_ptr<LinOp>>> blocks_;
};


}  // namespace matrix
}  // namespace gko

#endif  // GINKGO_BLOCK_MATRIX_HPP
