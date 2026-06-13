# Changelog

## 0.2.0

This release addresses the four known limits documented in 0.1.0. The autograd
engine now covers all operations used by the examples, training loops rely on
`ComputationGraph::backward_with_grad`, models can be saved and loaded as
plain-text checkpoints, datasets gain CSV import and split/shuffle utilities,
and a `Backend` trait lays architectural groundwork for future GPU backends.

Main changes:

- **Autograd** — Add gradient rules for Transpose, ReLU, Sigmoid, Tanh, and
  Softmax. The `local_gradients` method no longer returns `UnsupportedOperation`
  for any built-in op used by the examples. Finite-difference numerical
  verification tests confirm correctness for all five rules.
- **Training** — Rewrite `train_linear_regression`, `train_binary_classification`,
  `train_xor_mlp`, and `train_spiral_classifier` to build a `ComputationGraph`
  each epoch and call `backward_with_grad` for gradient propagation. Remove the
  hand-derived gradient formulas (~190 lines). Add `backward_with_grad` and
  `take_gradients` to `ComputationGraph`, and add `Tensor::row_add` with its
  `Operation::RowAdd` gradient rule.
- **Serialization** — New `serialize` module provides `save_linear` /
  `load_linear` and `save_xor_mlp` / `load_xor_mlp` using a human-readable
  line-oriented text format. CLI gains `--save-model PATH` on all three training
  commands. Auto-creates parent directories for convenience.
- **Data** — `Dataset::from_csv` parses CSV text with header and comment support.
  `Dataset::iter_rows` returns an `ExactSizeIterator` over `(features, targets)`
  pairs. `Dataset::shuffle(seed)` performs a deterministic Fisher-Yates shuffle
  using an LCG. `Dataset::split(ratio)` partitions into train/test subsets.
- **Backend** — New `backend` module defines a `Backend` trait covering add, sub,
  mul, div, matmul, transpose, sum, mean, sum_axis, mean_axis, and row_add.
  `CpuBackend` implements the trait as a zero-cost delegation to `Tensor`
  methods. The trait is object-safe, enabling future GPU backends without API
  changes.
- **Testing** — 43 new tests (11 autograd gradient + numerical, 9 serialization,
  13 data, 4 backend, 3 CLI integration, 3 existing CLI extensions). Clippy
  passes with zero warnings. Total test count: 292.

Known limits:

- GPU backend is architectural (trait + CpuBackend only); no actual GPU kernels;
- serialization covers `Linear` and `XorMlp`, not generic `Sequential`;
- no `--load-model` CLI flag for restoring weights before training;
- `ComputationGraph` is rebuilt each epoch — not reused across steps;
- softmax node in `train_spiral_classifier` does not participate in backward;
- all training examples use full-batch gradient descent.

## 0.1.0

RustGrad now has the main pieces needed for the course experiment. The project
contains a dense tensor type, a small reverse-mode autograd engine, neural
network layers, losses, optimizers, deterministic datasets, training loops, a
CLI, report export, and automated checks.

The implementation stays CPU-only and dependency-light. This keeps the code
small enough to read while still showing the path from tensor operations to a
working training command.

Main changes:

- tensor shape handling, indexing, reshape, arithmetic, reductions, transpose,
  and matrix multiplication;
- computation graph nodes, topological ordering, gradient storage, and backward
  rules for the operations used by the examples;
- `Linear`, `Sequential`, ReLU, Sigmoid, Tanh, and Softmax;
- mean squared error and cross entropy losses;
- SGD, Momentum, and Adam optimizers;
- deterministic linear regression, XOR, and spiral datasets;
- training loops for linear regression, binary classification, XOR MLP, and
  spiral softmax classification;
- Markdown and CSV report export through `--output DIR`;
- CLI commands: `train-linear`, `train-xor`, `train-spiral`, `inspect`;
- unit tests and CLI integration tests;
- GitHub Actions workflow for formatting, build, tests, and Clippy.

Known limits:

- no GPU backend;
- no large dataset loader;
- no serialization format for model checkpoints;
- autograd covers the operations needed by the current examples.

# 更新日志

## 0.2.0

本版本解决了 0.1.0 中记录的四个已知限制。自动求导引擎现已覆盖示例使用的所有
操作，训练循环基于 `ComputationGraph::backward_with_grad`，模型可以纯文本
checkpoint 格式保存和加载，数据集获得 CSV 导入和分割/打乱工具，`Backend`
trait 为未来的 GPU 后端奠定了架构基础。

主要变化：

- **自动求导** — 补全 Transpose、ReLU、Sigmoid、Tanh、Softmax 的梯度规则。
  `local_gradients` 对于示例使用的任何内置操作都不再返回 `UnsupportedOperation`。
  有限差分数值验证测试确认了全部五个规则的正确性。
- **训练** — 重写 `train_linear_regression`、`train_binary_classification`、
  `train_xor_mlp`、`train_spiral_classifier`，使其每轮构建 `ComputationGraph`
  并调用 `backward_with_grad` 进行梯度传播。移除手工推导的梯度公式（约 190 行）。
  为 `ComputationGraph` 新增 `backward_with_grad` 和 `take_gradients`，为
  `Tensor` 新增 `row_add` 及其 `Operation::RowAdd` 梯度规则。
- **序列化** — 新增 `serialize` 模块，提供 `save_linear` / `load_linear` 和
  `save_xor_mlp` / `load_xor_mlp`，采用人类可读的逐行文本格式。CLI 三个训练
  命令均支持 `--save-model PATH`。自动创建父目录。
- **数据** — `Dataset::from_csv` 解析支持表头和注释的 CSV 文本。
  `Dataset::iter_rows` 返回 `(features, targets)` 对的 `ExactSizeIterator`。
  `Dataset::shuffle(seed)` 使用 LCG 执行确定性 Fisher-Yates 洗牌。
  `Dataset::split(ratio)` 将数据集划分为训练/测试子集。
- **后端** — 新增 `backend` 模块，定义 `Backend` trait 覆盖 add、sub、mul、div、
  matmul、transpose、sum、mean、sum_axis、mean_axis、row_add。`CpuBackend`
  通过零成本委托到 `Tensor` 方法实现该 trait。trait 是 object-safe 的，未来
  可在不改变 API 的情况下接入 GPU 后端。
- **测试** — 新增 43 个测试（11 个 autograd 梯度 + 数值验证、9 个序列化、
  13 个数据、4 个后端、3 个 CLI 集成、3 个已有 CLI 扩展）。Clippy 零警告
  通过。总测试数：292。

已知限制：

- GPU 后端仅有 trait 接口，未实现实际 GPU kernel；
- 序列化覆盖 Linear 和 XorMlp，不含泛化的 Sequential 容器；
- CLI 尚未提供 `--load-model` 选项以在训练前恢复权重；
- `ComputationGraph` 每 epoch 从零重建，未跨步骤复用；
- `train_spiral_classifier` 的计算图中 Softmax 节点不参与反向传播；
- 所有训练示例均使用全批量梯度下降（full-batch GD）。

## 0.1.0

RustGrad 已经具备课程实验所需的主要内容。项目包含稠密张量类型、小型反向模式
自动求导引擎、神经网络层、损失函数、优化器、确定性数据集、训练循环、命令
行工具、报告导出和自动化检查。

当前实现保持 CPU-only，并尽量减少外部依赖。这样代码规模仍然适合阅读，同时
也能展示从张量运算到可运行训练命令的完整路径。

主要变化：

- 张量 shape 管理、索引、reshape、算术运算、归约、转置和矩阵乘法；
- 计算图节点、拓扑排序、梯度存储，以及示例所需操作的反向规则；
- `Linear`、`Sequential`、ReLU、Sigmoid、Tanh、Softmax；
- 均方误差和交叉熵损失；
- SGD、Momentum、Adam 优化器；
- 确定性的线性回归、XOR、螺旋数据集；
- 线性回归、二分类、XOR MLP、螺旋 softmax 分类训练循环；
- 通过 `--output DIR` 导出 Markdown 和 CSV 报告；
- CLI 命令：`train-linear`、`train-xor`、`train-spiral`、`inspect`；
- 单元测试和 CLI 集成测试；
- GitHub Actions 运行格式检查、构建、测试和 Clippy。

已知限制：

- 没有 GPU 后端；
- 没有大型数据集加载器；
- 没有模型 checkpoint 序列化格式；
- 自动求导覆盖当前示例需要的操作。
