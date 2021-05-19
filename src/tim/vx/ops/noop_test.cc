/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/noop.h"

#include "gtest/gtest.h"

TEST(OP, noop_shape_1_uint8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({1});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            io_shape, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data1 = { 255 };

    std::vector<uint8_t> golden = { 255 };

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(in_data1.data(), in_data1.size()));

    auto noop = graph->CreateOperation<tim::vx::ops::Noop>();
    (*noop).BindInputs({input_tensor1}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(1, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(OP, noop_shape_5_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({5});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = { -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };

    std::vector<float> golden = { -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(in_data1.data(), in_data1.size()*4));

    auto noop = graph->CreateOperation<tim::vx::ops::Noop>();
    (*noop).BindInputs({input_tensor1}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(5, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(OP, noop_shape_1_5_2_1_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({1,5,2,1,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = {
        -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity(),
        -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };

    std::vector<float> golden = {
        -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity(),
        -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(in_data1.data(), in_data1.size()*4));

    auto noop = graph->CreateOperation<tim::vx::ops::Noop>();
    (*noop).BindInputs({input_tensor1}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(10, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

