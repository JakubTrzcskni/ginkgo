// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/hbmc.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"

namespace {


class Hbmc : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = int;
    using reorder_type = gko::reorder::Hbmc<v_type, i_type>;
    using new_reorder_type = gko::experimental::reorder::Hbmc<i_type>;

    Hbmc()
        : exec(gko::ReferenceExecutor::create()),
          hbmc_factory(reorder_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<reorder_type::Factory> hbmc_factory;
};


TEST_F(Hbmc, HbmcFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->hbmc_factory->get_executor(), this->exec);
}


TEST_F(Hbmc, NewInterfaceDefaults)
{
    auto param = new_reorder_type::build();

    ASSERT_EQ(param.base_block_size, 4u);
    ASSERT_EQ(param.lvl_2_block_size,
              32u);
}


TEST_F(Hbmc, NewInterfaceSetParameters)
{
    auto param =
        new_reorder_type::build().with_base_block_size(8u).with_lvl_2_block_size(64u);

        ASSERT_EQ(param.base_block_size, 8u);
    ASSERT_EQ(param.lvl_2_block_size,
              64u);
}


}  // namespace
