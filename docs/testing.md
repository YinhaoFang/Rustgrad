# Testing Strategy

This document summarizes the testing strategy used by RustGrad.

## Goals

The test suite supports course grading and long-term maintenance:

- verify normal behavior for tensors, autograd, neural networks, losses,
  optimizers, data generation, training loops, reports, and CLI commands;
- cover boundary cases and invalid input paths;
- keep examples deterministic so failures are reproducible;
- run quickly on Windows and in GitHub Actions.

## Test Layers

RustGrad uses three layers of tests.

### Unit Tests

Most modules contain unit tests next to the code they verify. These tests focus
on small pieces of behavior:

- tensor shape validation, indexing, broadcasting, reductions, and matmul;
- autograd topological ordering and backward rules;
- activation functions and layer parameter shapes;
- loss values and loss validation errors;
- optimizer update rules and state handling;
- dataset shapes and deterministic values;
- training histories, metrics, and convergence checks;
- report formatting and file export.

Unit tests make failures easy to locate because each test usually maps to one
function or one small behavior.

### CLI Unit Tests

`src/main.rs` includes tests for command parsing and command output assembly.
These tests call the internal `execute` function directly, so they are fast and
can check detailed parser behavior:

- unknown command errors;
- missing option values;
- numeric option parsing;
- output format selection;
- report output paths.

### CLI Integration Tests

`tests/cli.rs` runs the compiled `rustgrad` binary through `Command`. These
tests verify behavior that unit tests cannot fully cover:

- process exit status;
- stdout for successful commands;
- stderr for error commands;
- actual report file creation through `--output`;
- machine-readable CSV output from the binary.

This gives confidence that examples work from a user's terminal and from Rust
function calls.

## Quality Commands

Run these commands before committing:

```bash
cargo fmt --check
cargo test
cargo clippy -- -D warnings
```

The same checks are also run by GitHub Actions.

## Current Coverage by Behavior

The test suite covers:

- shape mismatches and invalid arguments;
- deterministic dataset generation;
- tensor arithmetic and matrix operations;
- reverse-mode gradient propagation;
- trainable layer parameter updates;
- MSE and cross entropy losses;
- SGD, Momentum, and Adam behavior;
- linear regression convergence;
- binary classification convergence;
- XOR prediction;
- spiral classifier loss decrease and probability output;
- Markdown and CSV report generation;
- CLI success and failure paths.

## Reproducibility

RustGrad avoids random initialization in the course-project examples. Linear
layers use deterministic initialization, synthetic datasets are deterministic,
and training tests use fixed hyperparameters.

This reproducibility keeps the project easier to grade and helps make test
failures meaningful.

## Windows Notes

The test suite is expected to run directly on Windows with stable Rust. WSL2 is
not required for the current CPU-only implementation. Some Git or Cargo
commands may print line-ending or path-canonicalization warnings in local
Windows environments; these warnings do not indicate failed tests.

# 测试策略

本文总结 RustGrad 的测试策略。

## 目标

测试套件服务于课程评分和后续维护：

- 验证 tensor、autograd、神经网络、loss、optimizer、数据生成、训练循环、
  报告和 CLI 的正常行为；
- 覆盖边界情况和非法输入；
- 保持示例确定性，让失败可以复现；
- 保证测试能在 Windows 和 GitHub Actions 中快速运行。

## 测试层次

RustGrad 使用三层测试。

### 单元测试

大多数模块都把单元测试放在对应源码旁边。这些测试关注小范围行为：

- tensor 的 shape 校验、索引、广播、归约和矩阵乘法；
- autograd 的拓扑排序和反向传播规则；
- 激活函数和层参数形状；
- loss 数值和错误校验；
- optimizer 更新规则和状态管理；
- 数据集形状和确定性数值；
- 训练历史、指标和收敛检查；
- 报告格式化和文件导出。

单元测试让失败位置更容易定位，因为每个测试通常对应一个函数或一个小行为。

### CLI 单元测试

`src/main.rs` 包含命令解析和输出组装测试。这些测试直接调用内部 `execute`
函数，因此运行快，也能精确检查解析行为：

- 未知命令错误；
- 缺少参数值；
- 数值参数解析；
- 输出格式选择；
- 报告输出路径。

### CLI 集成测试

`tests/cli.rs` 通过 `Command` 运行编译后的 `rustgrad` 二进制文件。这些测试
验证单元测试无法完全覆盖的行为：

- 进程退出码；
- 成功命令的 stdout；
- 失败命令的 stderr；
- `--output` 是否真的创建报告文件；
- 二进制程序是否输出机器可读 CSV。

这能保证示例既能从用户终端真实运行，也能在 Rust 函数调用中稳定工作。

## 质量检查命令

提交前运行：

```bash
cargo fmt --check
cargo test
cargo clippy -- -D warnings
```

GitHub Actions 也会运行同样的检查。

## 当前行为覆盖

测试套件覆盖：

- shape 不匹配和非法参数；
- 确定性数据集生成；
- 张量算术和矩阵操作；
- 反向模式梯度传播；
- 可训练层参数更新；
- MSE 和交叉熵损失；
- SGD、Momentum、Adam 行为；
- 线性回归收敛；
- 二分类收敛；
- XOR 预测；
- 螺旋分类器 loss 下降和概率输出；
- Markdown 和 CSV 报告生成；
- CLI 成功和失败路径。

## 可复现性

RustGrad 在课程项目示例中避免随机初始化。线性层初始化是确定性的，合成数据
集是确定性的，训练测试也使用固定超参数。

这种可复现性让项目更容易评分，也让测试失败更有意义。

## Windows 说明

测试套件可以直接在 Windows 的稳定版 Rust 上运行。当前 CPU-only 实现不需要
WSL2。本地 Windows 环境下，部分 Git 或 Cargo 命令可能打印换行符或路径规范
化警告；这些警告不代表测试失败。
