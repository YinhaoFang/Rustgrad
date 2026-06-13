# Training Workflow

This document explains how RustGrad trains the example models used by the CLI
and tests.

## Training Data

RustGrad uses deterministic synthetic datasets so example output is stable
across machines:

- `linear_regression(samples, slope, intercept)` creates one-dimensional
  regression data in the interval `[-1, 1]`.
- `xor()` creates the classic four-row XOR classification dataset.
- `spiral(samples_per_class, classes)` creates a two-dimensional multi-class
  spiral dataset with one-hot targets.

Deterministic data is useful for grading because the same command should behave
consistently on Windows, Linux, CI, and the instructor's machine.

## Shared Training State

Training examples use a small set of shared types:

- `TrainingConfig`: epochs, learning rate, and logging interval.
- `TrainingRecord`: one epoch of loss and optional accuracy.
- `TrainingHistory`: an append-only list of training records.

The report and CLI modules consume `TrainingHistory`, which keeps training
loops independent from output formatting.

## Linear Regression

`train_linear_regression(dataset, config)` trains one `Linear` layer with mean
squared error and SGD.

The loop is intentionally explicit:

1. Run the linear layer: `prediction = input @ weights + bias`.
2. Compute MSE gradients for weights and bias.
3. Store gradients in a `GradientSet`.
4. Let the optimizer update model parameters.
5. Recompute predictions after the update.
6. Record the epoch loss.

The implementation supports multiple output columns, although the CLI example
uses a single output.

## Binary Classification

`train_binary_classification(dataset, config, threshold)` trains logistic
regression with binary cross entropy.

The target tensor must be a single-column matrix containing only `0.0` and
`1.0`. The loop records both loss and threshold-based accuracy, making
convergence easy to verify in tests and reports.

## XOR MLP

`train_xor_mlp(config)` trains a tiny two-layer sigmoid MLP specialized for the
XOR task:

```text
2 inputs -> 2 hidden sigmoid units -> 1 sigmoid output
```

The model starts from deterministic parameters that already represent a useful
XOR structure. Training still runs through the explicit gradient and optimizer
path, so the example demonstrates a non-linear network while staying stable and
fast.

CLI command:

```bash
cargo run -- train-xor --epochs 160 --learning-rate 0.4
```

## Spiral Softmax Classifier

`train_spiral_classifier(samples_per_class, classes, config)` trains a softmax
classifier on the deterministic spiral dataset.

The raw `(x, y)` spiral data is linearly inseparable. RustGrad applies a
compact polar feature map before the linear classifier:

```text
(x, y) -> radius and phase -> [cos(phase), sin(phase)]
```

This keeps the model simple while still showing an important machine-learning
idea: a non-linear feature transformation can make a problem easier for a
linear classification head.

CLI command:

```bash
cargo run -- train-spiral --epochs 160 --learning-rate 0.7 --samples-per-class 12 --classes 3
```

## Optimizer Flow

Training loops follow one update flow:

1. compute gradients;
2. collect mutable parameter references from the model;
3. call `optimizer.step(&mut parameters, &gradients)`.

SGD, Momentum, and Adam use the same interface. The current training examples
use SGD for readability, while the optimizer module tests verify the other
optimizers independently.

## Graph-Based Gradient Propagation

Starting from 0.2.0, all training loops build a `ComputationGraph` each epoch
and propagate gradients with `backward_with_grad`. The flow is:

1. Insert features as non-trainable leaf, weights and bias as trainable leaves.
2. Build forward operations: `MatMul → RowAdd → (optional activation)`.
3. Compute the combined loss gradient analytically (e.g. MSE:
   `(2/N)*(pred - target)`, sigmoid+BCE: `(σ(logits) - t)/N`, softmax+CE:
   `(softmax(logits) - targets)/N`).
4. Seed the output node with this loss gradient via `backward_with_grad`.
5. Take accumulated gradients from leaf nodes via `take_gradients`.
6. Pass gradients to the optimizer.

This approach combines the clarity of analytical loss gradients (easy to verify
in a report) with the generality of graph-based parameter gradient propagation
(correct through any activation or intermediate operation).

## Model Checkpoint

The CLI supports exporting trained model parameters:

```bash
cargo run -- train-linear --epochs 120 --save-model runs/linear.checkpoint
cargo run -- train-xor --epochs 160 --save-model runs/xor.checkpoint
```

Checkpoint files use a plain-text format: a header line with rank and
dimensions, followed by one `f64` value per line. The `serialize` module
provides `load_linear` and `load_xor_mlp` for programmatic reloading.

## Reports

The CLI can export report files with `--output DIR`:

```bash
cargo run -- train-spiral --epochs 160 --output runs/spiral-demo
```

The output directory contains:

- `summary.md`: Markdown summary and full history table.
- `history.csv`: epoch, loss, and accuracy columns for plotting.

These files can be used directly as training evidence in the experiment report.

# 训练流程

本文说明 RustGrad 如何训练 CLI 和测试中使用的示例模型。

## 训练数据

RustGrad 使用确定性的合成数据集，保证示例输出在不同机器上保持稳定：

- `linear_regression(samples, slope, intercept)`：在 `[-1, 1]` 区间生成一维
  回归数据。
- `xor()`：生成经典四行 XOR 分类数据集。
- `spiral(samples_per_class, classes)`：生成二维多分类螺旋数据集，目标为
  one-hot 编码。

确定性数据对课程评分很有帮助，因为同一条命令在 Windows、Linux、CI 和老师
的机器上应当表现一致。

## 共享训练状态

训练示例使用一组共享类型：

- `TrainingConfig`：训练轮数、学习率和日志间隔。
- `TrainingRecord`：单个 epoch 的 loss 和可选 accuracy。
- `TrainingHistory`：按顺序保存训练记录。

报告模块和 CLI 模块都消费 `TrainingHistory`，因此训练循环无需关心输出格式。

## 线性回归

`train_linear_regression(dataset, config)` 使用均方误差和 SGD 训练一个
`Linear` 层。

训练循环有意写得比较显式：

1. 执行线性层：`prediction = input @ weights + bias`。
2. 计算权重和偏置的 MSE 梯度。
3. 将梯度保存为 `GradientSet`。
4. 交给优化器更新模型参数。
5. 更新后重新计算预测。
6. 记录该 epoch 的 loss。

实现支持多输出列，虽然 CLI 示例使用的是单输出回归。

## 二分类

`train_binary_classification(dataset, config, threshold)` 使用二元交叉熵训练
逻辑回归模型。

目标张量必须是单列矩阵，且只包含 `0.0` 或 `1.0`。训练循环同时记录 loss 和
基于阈值的 accuracy，方便测试和报告验证收敛效果。

## XOR 多层感知机

`train_xor_mlp(config)` 训练一个专门用于 XOR 任务的小型两层 sigmoid MLP：

```text
2 inputs -> 2 hidden sigmoid units -> 1 sigmoid output
```

模型从确定性参数开始，这些参数已经表达了一个有用的 XOR 结构。训练仍然经过
显式梯度和优化器路径，因此示例既能展示非线性网络，又保持稳定、快速。

CLI 命令示例：

```bash
cargo run -- train-xor --epochs 160 --learning-rate 0.4
```

## 螺旋数据 Softmax 分类器

`train_spiral_classifier(samples_per_class, classes, config)` 在确定性的螺旋
数据集上训练 softmax 分类器。

原始 `(x, y)` 螺旋数据线性不可分。RustGrad 在线性分类器之前应用一个紧凑的
极坐标特征映射：

```text
(x, y) -> radius and phase -> [cos(phase), sin(phase)]
```

这样模型仍然足够简单，同时展示了一个重要机器学习思想：非线性特征变换可以
让线性分类头更容易解决问题。

CLI 命令示例：

```bash
cargo run -- train-spiral --epochs 160 --learning-rate 0.7 --samples-per-class 12 --classes 3
```

## 优化器流程

训练循环遵循统一更新流程：

1. 计算梯度；
2. 从模型中收集可变参数引用；
3. 调用 `optimizer.step(&mut parameters, &gradients)`。

SGD、Momentum 和 Adam 都使用同样接口。当前训练示例为了可读性使用 SGD，其
他优化器通过 `optim` 模块测试单独验证。

## 基于计算图的梯度传播

从 0.2.0 开始，所有训练循环每 epoch 构建一次 `ComputationGraph` 并通过
`backward_with_grad` 传播梯度。流程如下：

1. 将特征作为不可训练叶节点、权重和偏置作为可训练叶节点插入。
2. 构建前向操作链：`MatMul → RowAdd → （可选激活函数）`。
3. 解析计算组合损失梯度（如 MSE：`(2/N)*(pred - target)`，sigmoid+BCE：
   `(σ(logits) - t)/N`，softmax+CE：`(softmax(logits) - targets)/N`）。
4. 将损失梯度作为种子传入 `backward_with_grad`。
5. 通过 `take_gradients` 取出叶节点的累积梯度。
6. 将梯度交给优化器。

这种方案结合了解析损失梯度的清晰性（便于在报告中验证）和计算图梯度传播的
通用性（可正确处理任意激活函数和中间操作）。

## 模型保存

CLI 支持导出训练好的模型参数：

```bash
cargo run -- train-linear --epochs 120 --save-model runs/linear.checkpoint
cargo run -- train-xor --epochs 160 --save-model runs/xor.checkpoint
```

Checkpoint 文件使用纯文本格式：rank,dims 标头行 + 每行一个 f64 值。
`serialize` 模块提供 `load_linear` 和 `load_xor_mlp` 供编程加载。

## 报告导出

CLI 可以通过 `--output DIR` 导出报告文件：

```bash
cargo run -- train-spiral --epochs 160 --output runs/spiral-demo
```

输出目录包含：

- `summary.md`：Markdown 摘要和完整训练历史表。
- `history.csv`：用于绘图的 epoch、loss、accuracy 数据。

这些文件可以直接作为实验报告的训练证据。
