# RustGrad Technical Report

RustGrad is a Rust course project that implements the core path of a small deep
learning framework. The project starts from dense tensor operations, builds a
small reverse-mode automatic differentiation engine, adds neural-network
building blocks and optimizers, then exposes runnable training examples through
a command-line interface.

Module-level details are documented separately in `docs/autograd.md`,
`docs/training.md`, and `docs/testing.md`. This report focuses on how those
parts fit together as a complete experiment.

The scope is intentionally controlled. The project stays CPU-only and uses
deterministic synthetic datasets. This keeps the implementation readable while
still covering the important contracts in a training system: tensor shapes must
be checked, graph dependencies must be recorded, gradients must be accumulated
correctly, optimizers must update parameters in a stable order, and command-line
examples must produce reproducible output.

The code is organized by responsibility:

- `tensor`: dense data storage, shape metadata, indexing, reshape, arithmetic,
  reductions, transpose, and matrix multiplication;
- `autograd`: computation graph nodes, operation metadata, topological ordering,
  gradient storage, and backward propagation;
- `nn`: `Linear`, `Sequential`, activation functions, and the `Module` trait;
- `loss`: mean squared error and cross entropy;
- `optim`: SGD, Momentum, and Adam;
- `data`: deterministic synthetic datasets;
- `train`: training configuration, metrics, histories, and example training
  loops;
- `report`: Markdown and CSV export;
- `main.rs`: CLI commands for training and inspection.

## Tensor Layer

The tensor module provides the basic numerical type used by the rest of the
project. A `Tensor` stores values in row-major order and carries explicit shape
metadata. Constructors validate empty shapes, zero dimensions, and mismatched
data lengths. Indexing supports both flat access and multidimensional access,
which makes tests and higher-level modules easier to write.

The implemented operations cover the needs of the framework examples:
elementwise add, subtraction, multiplication, division, scalar-like
broadcasting, transpose, reshape, sum, mean, axis reductions, and matrix
multiplication. The shape checks are part of the technical design, not only
defensive code. Later modules rely on these checks to reject invalid training
inputs before they can produce misleading results.

Matrix multiplication is especially important because it is the foundation of
the `Linear` layer. The implementation checks that both operands are rank-2
matrices and that the left column count equals the right row count. Errors are
reported through `RustGradError`, which keeps failure handling consistent across
the project.

## Automatic Differentiation

The autograd module implements a small reverse-mode computation graph. Each
`GraphNode` stores a tensor value, a list of parent nodes, the operation that
created the value, a `requires_grad` flag, and an optional accumulated gradient.
`NodeId` values are stable indices into the graph.

Forward operations create new nodes with parent links. During backward
propagation, `ComputationGraph::backward(output)` validates the output node,
computes a topological order, clears previous gradients, seeds the output with
an all-ones gradient, then walks the graph in reverse. Each operation applies
its own gradient rule and accumulates gradients into parents that require them.

The engine currently supports backward rules for add, sub, mul, div, matmul,
sum, and mean. These operations are enough to demonstrate the main mechanics of
reverse-mode differentiation and to test the graph behavior with scalars,
vectors, matrices, repeated parents, and broadcast-like reductions.

Gradient accumulation is one of the key details. If a node contributes to the
output through more than one path, its gradient must be the sum of all incoming
contributions. The tests cover repeated parents and multiple downstream paths
because this is a common source of incorrect autograd implementations.

Gradients are stored on graph nodes instead of inside `Tensor`. This separation
keeps the tensor type focused on dense numerical data. It also makes the
autograd engine easier to explain: tensor values, graph nodes, and gradients are
separate concepts with separate responsibilities.

## Neural Network Components

The `nn` module defines a `Module` trait with `forward`, `parameters_mut`, and
`name`. `Linear` implements the standard affine transform:

```text
output = input @ weights + bias
```

Weights have shape `[input_size, output_size]`, and bias has shape
`[output_size]`. The layer supports vector and matrix input, validates feature
counts, and exposes mutable parameter references for optimizers.

The module also includes `Sequential`, which applies child modules in order,
and common activations: ReLU, Sigmoid, Tanh, and Softmax. Softmax is implemented
with a numerically stable row-wise normalization for matrix inputs, which
matches batched multi-class classification.

## Losses and Optimizers

The loss module contains mean squared error and cross entropy. MSE is used for
linear regression. Cross entropy takes logits and one-hot or distribution-like
targets, applies a stable log-sum-exp form internally, and returns an average
loss across rows for matrix input.

The optimizer module contains `GradientSet` and three update rules:

- SGD: direct gradient descent;
- Momentum: velocity accumulation with a momentum coefficient;
- Adam: first and second moment estimates with bias correction.

Training examples currently use SGD because its update rule is easiest to
inspect in a report. Momentum and Adam are still implemented and tested
independently, which shows that the optimizer abstraction is not tied to a
single update rule.

## Training Implementations

The training module provides shared types:

- `TrainingConfig`: epochs, learning rate, and logging interval;
- `TrainingRecord`: one epoch of loss and optional accuracy;
- `TrainingHistory`: a sequence of records used by reports and CLI output.

`train_linear_regression` trains a single `Linear` layer with MSE and SGD. The
gradient formulas are written directly for the affine layer. This makes the
training loop easy to inspect: compute predictions, compute weight and bias
gradients, update parameters, recompute loss, record history.

`train_binary_classification` trains logistic regression with binary cross
entropy. Targets must be one-column matrices containing `0.0` and `1.0`. The
loop records threshold accuracy after each update.

`train_xor_mlp` trains a small 2-2-1 sigmoid MLP on the XOR dataset. The model
uses deterministic initial parameters and explicit backpropagation formulas for
the two-layer network. This example shows a non-linear model while keeping the
state small enough to inspect.

`train_spiral_classifier` trains a softmax classifier on deterministic spiral
data. Raw spiral coordinates are linearly inseparable, so the training code
maps each point into a compact polar feature representation:

```text
(x, y) -> radius and phase -> [cos(phase), sin(phase)]
```

The classifier then trains a linear softmax head with cross entropy. This
example is useful because it demonstrates a practical idea with a small amount
of code: feature transformations can change the difficulty of a classification
problem.

## CLI and Reports

The CLI exposes the training examples as runnable commands:

```bash
cargo run -- train-linear --epochs 120 --learning-rate 0.12 --samples 31
cargo run -- train-xor --epochs 160 --learning-rate 0.4
cargo run -- train-spiral --epochs 160 --learning-rate 0.7 --samples-per-class 12 --classes 3
cargo run -- inspect
```

Each training command supports `--format text|csv|markdown`. It can also export
report files:

```bash
cargo run -- train-spiral --epochs 160 --output runs/spiral-demo
```

The output directory contains `summary.md` and `history.csv`. The Markdown file
records summary metrics and a training history table. The CSV file records
epoch, loss, and optional accuracy, which is useful for plotting loss curves or
attaching raw training evidence.

The CLI parser is implemented with the standard library. This avoids an extra
dependency and keeps argument handling visible. Error paths are tested for
unknown commands and missing option values.

## Verification

The project is verified at three levels. Unit tests check module-level behavior:
tensor math, shape errors, autograd gradient rules, neural network layers,
losses, optimizers, datasets, training convergence, and report generation. CLI
unit tests call the internal command executor and check parsing and output
assembly. CLI integration tests run the compiled `rustgrad` binary and check
exit status, stdout, stderr, CSV output, and report file creation.

At the time of this report, local checks pass with:

- 232 library tests;
- 11 CLI unit tests;
- 6 CLI integration tests.

The quality commands are:

```bash
cargo fmt --check
cargo test
cargo clippy -- -D warnings
```

GitHub Actions runs formatting, build, tests, and Clippy on push and pull
request events. The project also runs directly on Windows with stable Rust.
The current implementation does not require WSL2.

## Limits

RustGrad has no GPU backend, no large dataset pipeline, and no model checkpoint
format. The autograd engine covers the operations needed by the current
examples. The training loops use small deterministic datasets and explicit
gradient formulas. These choices keep the project suitable for a course
experiment and make the behavior easier to verify.

The most important technical result is a complete, testable path from tensor
operations to model training and report export. Each layer is small, but the
interfaces between layers are real: tensors enforce shape rules, graph nodes
carry dependency information, optimizers update parameter tensors, and CLI
commands exercise the same code paths as tests.

# RustGrad 技术报告

RustGrad 是一个 Rust 课程项目，实现了一个小型深度学习框架的核心路径。项目
从稠密张量运算开始，构建小型反向模式自动求导引擎，加入神经网络组件和优化
器，并通过命令行工具提供可运行的训练示例。

更细的模块说明见 `docs/autograd.md`、`docs/training.md` 和
`docs/testing.md`。本报告关注这些部分如何组成一个完整实验。

项目范围经过控制。当前实现保持 CPU-only，并使用确定性的合成数据集。这样代
码仍然适合阅读，同时覆盖训练系统中的关键约定：张量 shape 需要被检查，计算
图依赖需要被记录，梯度需要正确累加，优化器需要按稳定顺序更新参数，命令行
示例需要产生可复现输出。

代码按职责组织：

- `tensor`：稠密数据存储、shape 元数据、索引、reshape、算术运算、归约、
  转置和矩阵乘法；
- `autograd`：计算图节点、操作元数据、拓扑排序、梯度存储和反向传播；
- `nn`：`Linear`、`Sequential`、激活函数和 `Module` trait；
- `loss`：均方误差和交叉熵；
- `optim`：SGD、Momentum、Adam；
- `data`：确定性合成数据集；
- `train`：训练配置、指标、历史记录和示例训练循环；
- `report`：Markdown 和 CSV 导出；
- `main.rs`：训练与检查命令行入口。

## 张量层

张量模块提供项目其余部分使用的基础数值类型。`Tensor` 按行优先顺序保存数
据，并带有显式 shape 元数据。构造函数会检查空 shape、零维度和数据长度不匹
配。索引支持 flat 访问和多维访问，这让测试和上层模块都更容易编写。

已实现操作覆盖当前框架示例的需要：逐元素加、减、乘、除，类标量广播，转
置，reshape，求和，均值，按轴归约，以及矩阵乘法。shape 检查是技术设计的
一部分。后续模块依赖这些检查，在训练输入无效时尽早返回错误，减少误导性结
果。

矩阵乘法尤其重要，因为它是 `Linear` 层的基础。实现会检查两个输入都是
rank-2 矩阵，并检查左矩阵列数等于右矩阵行数。错误通过 `RustGradError` 报
告，因此项目中的失败处理风格保持一致。

## 自动求导

自动求导模块实现了一个小型反向模式计算图。每个 `GraphNode` 保存张量值、父
节点列表、产生该值的操作、`requires_grad` 标记，以及可选的累积梯度。
`NodeId` 是图中的稳定索引。

前向操作会创建带父节点链接的新节点。执行反向传播时，
`ComputationGraph::backward(output)` 会检查输出节点，计算拓扑顺序，清空旧
梯度，用全 1 初始化输出梯度，再按反向顺序遍历计算图。每个操作应用自己的
梯度规则，并把梯度累加到需要梯度的父节点上。

当前引擎支持 add、sub、mul、div、matmul、sum 和 mean 的反向规则。这些操作
足以展示反向模式求导的主要机制，也能用标量、向量、矩阵、重复父节点和类广
播归约来测试计算图行为。

梯度累加是关键细节之一。如果某个节点通过多条路径影响输出，它的梯度必须是
所有路径贡献之和。测试覆盖了重复父节点和多条下游路径，因为这是自动求导实
现中常见的错误来源。

梯度保存在计算图节点上，没有放入 `Tensor` 内部。这样可以让 `Tensor` 专注
于稠密数值数据，也让自动求导引擎更容易说明：张量值、计算图节点和梯度分别
承担不同职责。

## 神经网络组件

`nn` 模块定义了 `Module` trait，包含 `forward`、`parameters_mut` 和
`name`。`Linear` 实现标准仿射变换：

```text
output = input @ weights + bias
```

权重形状为 `[input_size, output_size]`，偏置形状为 `[output_size]`。该层支
持向量和矩阵输入，会校验特征数量，并向优化器暴露可变参数引用。

模块还包含 `Sequential`，用于按顺序应用子模块，以及 ReLU、Sigmoid、Tanh、
Softmax 等常见激活函数。Softmax 对矩阵输入执行数值稳定的逐行归一化，符合
批量多分类场景。

## 损失函数和优化器

损失模块包含均方误差和交叉熵。MSE 用于线性回归。交叉熵接收 logits 和
one-hot 或分布形式的目标，内部使用稳定的 log-sum-exp 形式，并在矩阵输入时
返回逐行平均 loss。

优化器模块包含 `GradientSet` 和三种更新规则：

- SGD：直接梯度下降；
- Momentum：带动量系数的速度累积；
- Adam：一阶、二阶矩估计和偏差校正。

训练示例目前使用 SGD，因为它的更新规则最容易在报告中检查。Momentum 和
Adam 仍然被实现并单独测试，这说明优化器抽象没有绑定到单一更新规则。

## 训练实现

训练模块提供共享类型：

- `TrainingConfig`：训练轮数、学习率和日志间隔；
- `TrainingRecord`：单个 epoch 的 loss 和可选 accuracy；
- `TrainingHistory`：用于报告和 CLI 输出的记录序列。

`train_linear_regression` 使用 MSE 和 SGD 训练单个 `Linear` 层。仿射层的梯
度公式直接写在代码中。训练循环因此比较容易检查：计算预测，计算权重和偏置
梯度，更新参数，重新计算 loss，记录历史。

`train_binary_classification` 使用二元交叉熵训练逻辑回归。目标必须是只包含
`0.0` 和 `1.0` 的单列矩阵。循环会在每次更新后记录阈值 accuracy。

`train_xor_mlp` 在 XOR 数据集上训练一个小型 2-2-1 sigmoid MLP。模型使用确
定性初始参数，并为两层网络显式计算反向传播公式。这个示例展示了非线性模型，
同时状态规模仍然适合检查。

`train_spiral_classifier` 在确定性螺旋数据上训练 softmax 分类器。原始螺旋坐
标线性不可分，因此训练代码把每个点映射到紧凑的极坐标特征表示：

```text
(x, y) -> radius and phase -> [cos(phase), sin(phase)]
```

分类器随后使用交叉熵训练线性 softmax 头。这个示例有实际意义，因为它用较少
代码展示了一个常见思想：特征变换会改变分类问题的难度。

## CLI 和报告

CLI 将训练示例暴露为可运行命令：

```bash
cargo run -- train-linear --epochs 120 --learning-rate 0.12 --samples 31
cargo run -- train-xor --epochs 160 --learning-rate 0.4
cargo run -- train-spiral --epochs 160 --learning-rate 0.7 --samples-per-class 12 --classes 3
cargo run -- inspect
```

每个训练命令支持 `--format text|csv|markdown`。命令也可以导出报告文件：

```bash
cargo run -- train-spiral --epochs 160 --output runs/spiral-demo
```

输出目录包含 `summary.md` 和 `history.csv`。Markdown 文件记录汇总指标和训
练历史表。CSV 文件记录 epoch、loss 和可选 accuracy，便于绘制 loss 曲线或
附加原始训练证据。

CLI 解析使用标准库实现。这样避免额外依赖，并让参数处理过程可见。未知命令
和缺少参数值等错误路径都有测试覆盖。

## 验证

项目分三层验证。单元测试检查模块级行为：张量计算、shape 错误、自动求导规
则、神经网络层、损失函数、优化器、数据集、训练收敛和报告生成。CLI 单元测
试调用内部命令执行函数，检查解析和输出组装。CLI 集成测试运行编译后的
`rustgrad` 二进制文件，检查退出码、stdout、stderr、CSV 输出和报告文件创建。

写作本报告时，本地检查通过了：

- 232 个库测试；
- 11 个 CLI 单元测试；
- 6 个 CLI 集成测试。

质量检查命令如下：

```bash
cargo fmt --check
cargo test
cargo clippy -- -D warnings
```

GitHub Actions 会在 push 和 pull request 时运行格式检查、构建、测试和
Clippy。项目可以直接在 Windows 的稳定版 Rust 上运行。当前实现不需要 WSL2。

## 限制

RustGrad 没有 GPU 后端，没有大型数据集流水线，也没有模型 checkpoint 格式。
自动求导引擎覆盖的是当前示例需要的操作。训练循环使用小型确定性数据集和显
式梯度公式。这些选择让项目适合作为课程实验，也让行为更容易验证。

最重要的技术结果是一条完整、可测试的路径：从张量操作到模型训练，再到报告
导出。每一层都不大，但层与层之间的接口是真实的：张量维护 shape 规则，计
算图节点携带依赖信息，优化器更新参数张量，CLI 命令执行与测试相同的代码路
径。
