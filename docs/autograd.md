# Autograd Design

This document explains the automatic differentiation module in RustGrad. The
focus is to make the core ideas of reverse-mode autograd visible in a Rust
course project.

## Core Types

The autograd module is organized around three concepts:

- `NodeId`: a stable identifier for a node in the computation graph.
- `GraphNode`: the value, gradient, parent links, and producing operation.
- `ComputationGraph`: an append-only graph that stores nodes and runs backward
  propagation.

Each graph node stores a `Tensor` value. If the node is trainable or needs a
gradient, `requires_grad` is set to `true`. During `backward`, gradients are
stored on graph nodes outside the main `Tensor` type. This keeps tensor math
and graph bookkeeping easy to inspect separately.

## Operation Model

The `Operation` enum records how a node was created. All built-in operations
used by the training examples have complete backward rules:

- addition, subtraction, multiplication, division
- matrix multiplication
- sum, mean
- transpose
- ReLU, Sigmoid, Tanh, Softmax
- row-wise bias addition (RowAdd)

Gradient rules for activation functions (Sigmoid, Tanh) use output values
rather than input values, avoiding redundant exponential evaluation.
Softmax uses the vector-Jacobian product formulation
`grad[i] = s[i] * (g[i] - Σ_j s[j] * g[j])`.

Unsupported operations still return a clear `UnsupportedOperation` error,
making missing gradient rules explicit and helping prevent silent training
mistakes.

## Forward Pass

A forward pass creates nodes in dependency order:

1. Insert leaf tensors with `add_leaf`.
2. Insert operation nodes with parent `NodeId` values.
3. Store the output tensor and parent links for each operation.

The graph keeps the implementation direct and readable. Each operation maps
closely to formulas covered by unit tests.

## Backward Pass

`ComputationGraph::backward(output)` performs reverse-mode differentiation:

1. Validate that the output node exists.
2. Compute a topological order from dependencies to output.
3. Clear gradients from previous backward calls.
4. Seed the output node with an all-ones gradient.
5. Walk the topological order in reverse.
6. Apply the matching backward rule for each operation.
7. Accumulate gradients into parent nodes that require gradients.

Gradient accumulation matters when a value is reused by multiple downstream
operations. The tests cover repeated parents and multiple paths to ensure that
gradients are added correctly.

## Shape Handling

The autograd engine relies on the tensor module for shape validation and tensor
math. Backward rules also handle operation-specific shape behavior:

- elementwise operations produce elementwise gradients;
- scalar-like broadcasting is reduced back to the scalar parent;
- matrix multiplication uses standard matrix derivative formulas;
- `sum` and `mean` expand scalar output gradients back to the input shape.

Shape errors use the shared `RustGradError` type, keeping failures consistent
across tensors, autograd, training, and CLI commands.

## Gradient Storage

In the current course-project version, RustGrad stores graph gradients on graph
nodes instead of placing them inside `Tensor`. This has three advantages:

- `Tensor` remains a compact dense array type.
- The computation graph is easier to print, inspect, and test.
- Optimizers can be demonstrated separately through explicit `GradientSet`
  values.

This separation also makes it easier to explain the difference between a tensor
value, a graph node, and an accumulated derivative.

## Important Tests

The autograd tests cover both normal and edge cases:

- topological ordering visits dependencies before outputs;
- old gradients are cleared before a new backward pass;
- gradients accumulate from repeated parents;
- scalar add, sub, mul, and div gradients match hand calculations;
- vector elementwise gradients are correct;
- matrix multiplication gradients are correct;
- `sum` and `mean` gradients expand to input shapes;
- unsupported operations return clear errors.

These tests are intentionally small and explicit, making each derivative rule
auditable in a course report.

# 自动求导设计

本文说明 RustGrad 的自动求导模块，重点是让反向模式自动求导的核心思想在
Rust 课程项目中清晰可见。

## 核心类型

自动求导模块围绕三个核心概念组织：

- `NodeId`：计算图节点的稳定标识。
- `GraphNode`：保存节点值、梯度、父节点以及产生该节点的操作。
- `ComputationGraph`：追加式计算图，负责保存节点并执行反向传播。

每个图节点都保存一个 `Tensor` 值。如果节点可训练或需要梯度，
`requires_grad` 会被设为 `true`。执行 `backward` 时，梯度保存在对应的图
节点上，而非直接放入主 `Tensor` 类型中。这样可以把张量计算和计算图管理分
开，便于阅读、测试和报告说明。

## 操作模型

`Operation` 枚举记录节点的产生方式。所有训练示例使用的内置操作均有完整的
反向规则：

- 加法、减法、乘法、除法
- 矩阵乘法
- 求和、均值
- 转置
- ReLU、Sigmoid、Tanh、Softmax
- 行方向偏置加法（RowAdd）

激活函数（Sigmoid、Tanh）的梯度规则使用输出值而非输入值计算导数，避免重复
指数运算。Softmax 使用向量-雅可比积公式
`grad[i] = s[i] * (g[i] - Σ_j s[j] * g[j])`。

不支持的操作会返回清晰的 `UnsupportedOperation` 错误。这样可以显式暴露缺失
的梯度规则，减少静默训练错误。

## 前向传播

前向传播按依赖顺序创建节点：

1. 使用 `add_leaf` 插入叶子张量。
2. 使用父节点 `NodeId` 插入操作节点。
3. 每个操作节点保存输出张量和父节点链接。

计算图实现保持直接、可读。每个操作都尽量与单元测试中的数学公式对应。

## 反向传播

`ComputationGraph::backward(output)` 执行反向模式自动求导：

1. 检查输出节点是否存在。
2. 计算从依赖节点到输出节点的拓扑顺序。
3. 清空上一次反向传播留下的旧梯度。
4. 用全 1 梯度初始化输出节点。
5. 按拓扑顺序反向遍历。
6. 根据每个操作应用对应的反向规则。
7. 将梯度累加到需要梯度的父节点上。

当同一个值被多个后续操作复用时，梯度累加非常重要。测试覆盖了重复父节点和
多路径传播，确保梯度能正确相加。

## 形状处理

自动求导引擎依赖 `tensor` 模块进行形状校验和基础张量计算。反向规则也会处
理各类操作引入的形状关系：

- 逐元素操作产生逐元素梯度；
- 类标量广播会把梯度归约回标量父节点；
- 矩阵乘法使用标准矩阵求导公式；
- `sum` 和 `mean` 会把标量输出梯度扩展回输入形状。

形状错误统一使用 `RustGradError`，因此 tensor、autograd、training 和 CLI 的
错误风格保持一致。

## 梯度存储

RustGrad 在当前课程项目版本中把图梯度保存在计算图节点里，未直接放入
`Tensor`。这样做有三个好处：

- `Tensor` 保持为紧凑的稠密数组类型。
- 计算图更容易打印、检查和测试。
- 优化器可以通过显式的 `GradientSet` 单独演示。

这种分离也便于说明张量值、计算图节点和累积导数之间的区别。

## 关键测试

自动求导测试覆盖正常路径和边界情况：

- 拓扑排序保证依赖节点先于输出节点；
- 新一轮反向传播前会清空旧梯度；
- 重复父节点的梯度会正确累加；
- 标量加、减、乘、除的梯度与手算一致；
- 向量逐元素梯度正确；
- 矩阵乘法梯度正确；
- `sum` 和 `mean` 的梯度能扩展回输入形状；
- 不支持的操作会返回清晰错误。

这些测试故意保持小而明确，便于在课程报告中逐条解释每个求导规则。
