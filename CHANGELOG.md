# Changelog

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
