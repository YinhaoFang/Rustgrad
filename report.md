# RustGrad 项目报告

## 一、项目概述

RustGrad 是一个用 Rust 实现的面向教学的深度学习框架。项目的核心目标是：**从
零开始，不依赖任何外部深度学习库，用 Rust 构建一个麻雀虽小五脏俱全的训练系
统**——覆盖张量运算、自动求导、神经网络层、损失函数、优化器、数据集管理、训
练循环、模型序列化、命令行工具和报告导出。

### 项目定位

- **不是** PyTorch/TensorFlow 的 Rust 绑定或 wrapper。
- **是**一个完全自包含的教学框架，所有核心算法（矩阵乘法、反向传播、梯度下降
  及其变体）都在本仓库中手写实现。
- **目标用户**：课程评分老师（通过代码审查和 CLI 演示验证）以及未来的学习者
  （通过阅读源码理解深度学习框架的内部原理）。

### 技术约束

- 零外部运行时依赖（`Cargo.toml` 中 `[dependencies]` 为空）。
- CPU-only，使用 `f64` 精度。
- 纯文本模型序列化，不引入 serde/JSON 等依赖。
- CLI 参数解析使用标准库手写，不引入 clap。
- 跨平台兼容（Windows / Linux / macOS），已通过 GitHub Actions CI 验证。

---

## 二、项目结构

```
src/
  tensor/mod.rs     (1155 行)  稠密张量：shape、索引、算术、广播、归约、矩阵乘法、row-add
  autograd/mod.rs   (1760 行)  计算图、拓扑排序、12 种梯度规则、backward_with_grad、数值验证
  nn/mod.rs         ( 750 行)  Linear、Sequential、ReLU/Sigmoid/Tanh/Softmax 及导数
  loss/mod.rs       ( 400 行)  MSE、CrossEntropy（数值稳定 log-sum-exp）
  optim/mod.rs      (1180 行)  SGD、Momentum（速度累积）、Adam（偏差校正矩估计）
  data/mod.rs       ( 820 行)  合成数据集（线性/XOR/螺旋）、CSV 加载、shuffle、split
  train/mod.rs      (1085 行)  训练配置/指标/历史、4 个图式训练循环
  serialize.rs      ( 310 行)  纯文本模型 checkpoint 序列化/反序列化
  backend.rs        ( 180 行)  Backend trait + CpuBackend（GPU 架构预留）
  report/mod.rs     ( 510 行)  Markdown/CSV 报告导出、文件包写入
  error.rs          ( 150 行)  统一错误类型体系
  main.rs           ( 750 行)  CLI：train-linear/train-xor/train-spiral/inspect
  lib.rs            (  35 行)  模块注册和版本号
tests/
  cli.rs            ( 210 行)  端到端集成测试（二进制级别）
docs/               (4 篇)    autograd 设计、训练流程、测试策略、实验报告（中英双语）
```

---

## 三、各模块详解

### 3.1 张量系统（Tensor）

**设计思路**：`Tensor` 由 `Shape`（维度元数据）和 `Vec<f64>`（行优先稠密存
储）组成。不使用泛型和 trait 约束保持类型简单，降低教学复杂度。

**核心能力**：

| 功能 | 实现 | 说明 |
|------|------|------|
| 构造与校验 | `new`/`scalar`/`vector`/`matrix`/`zeros`/`ones`/`full` | 拒绝空 shape、零维度、数据长度不匹配 |
| 多维索引 | `get(&[usize])` / `set(&[usize])` | 行优先偏移计算，含边界检查 |
| 逐元素运算 | `add`/`sub`/`mul`/`div` | 同形直接运算 + 标量广播 |
| 矩阵乘法 | `matmul` | 三层循环实现，验证 rank-2 和内维匹配 |
| 转置 | `transpose` | 仅支持矩阵，逐元素重排 |
| reshape/flatten | `reshape`/`flatten` | 校验 element_count 匹配 |
| 归约 | `sum`/`mean`/`sum_axis`/`mean_axis` | 全量归约 + 按轴归约（向量/矩阵） |
| 行偏置加法 | `row_add` | 将向量广播叠加到矩阵每一行 |

**设计决策**：`Tensor` 本身不携带梯度信息。梯度存储在计算图节点上，二者分离
使得张量运算和自动求导各司其职。

### 3.2 自动求导系统（Autograd）

**架构**：`ComputationGraph` 是一个追加式的有向无环图。`NodeId` 是图中节点的
稳定索引。每次前向操作创建新节点并记录父节点链接和操作类型。反向传播时按拓
扑逆序遍历，对每个操作调用对应的梯度规则。

**梯度规则（12 种）**：

| 操作 | 梯度规则 | 方向 |
|------|---------|------|
| Add | `(g, g)` 直接传递 | 二元 |
| Sub | `(g, -g)` 右操作数取负 | 二元 |
| Mul | `(g*y, g*x)` 交叉乘积 | 二元 |
| Div | `(g/y, -g*x/y²)` x/y 的导数 | 二元 |
| MatMul | `(g @ Bᵀ, Aᵀ @ g)` 矩阵乘积链 | 二元 |
| Sum | 全 1 广播回原形 | 一元 |
| Mean | `1/N` 广播回原形 | 一元 |
| RowAdd | `(g, sum_axis_0(g))` 行求和 | 二元 |
| Transpose | `gᵀ` 转置回原形 | 一元 |
| ReLU | `g * I(x>0)` 指示函数 | 一元 |
| Sigmoid | `g * σ * (1-σ)` 用输出值 | 一元 |
| Tanh | `g * (1-tanh²)` 用输出值 | 一元 |
| Softmax | `g[i] = s[i] * (g_raw[i] - Σⱼ s[j]*g_raw[j])` | 一元 |

**关键接口**：

- `backward(output)` — 传统接口，以全 1 梯度种子输出节点。
- `backward_with_grad(output, seed)` — **训练专用接口**：训练循环解析计算损失
  对 logits/predictions 的梯度，作为种子传入，由 autograd 引擎继续反向传播到
  参数节点。这分离了"损失函数梯度"（手动，易于验证）和"参数梯度"（自动，覆
  盖复杂前向路径）。
- `take_gradients()` — 按插入顺序取出所有 `requires_grad=true` 的叶子节点梯
  度，直接供优化器使用。

**验证方法**：
- 解析验证：每个梯度规则都有对应的单元测试，将 autograd 输出与手算公式逐元
  素对比。
- 有限差分验证：`check_gradient_numeric` 框架对每个叶节点值施加 ±ε 扰动，计
  算 `(f(x+ε) - f(x-ε)) / 2ε`，与 autograd 梯度比较（容差 1e-5）。需要重建
  整个计算图以适应扰动后的值，每轮仅扰动一个元素保持精确。

### 3.3 神经网络层

**`Module` trait**：统一的神经网络组件接口。

```rust
pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn name(&self) -> &str;
}
```

**`Linear` 层**：`output = input @ weights + bias`。Weights 形状
`[input_size, output_size]`，bias 形状 `[output_size]`。支持向量和矩阵输入（
向量内部 reshape 为 1×N 矩阵处理）。构造函数使用确定性权重初始化（基于输入/
输出维度的对称模式），保证跨机器可复现。

**`Sequential` 容器**：按插入顺序依次应用子模块。通过 `Box<dyn Module>` 支持
异构层组合。`parameters_mut` 递归收集所有子模块的可训练参数，使优化器无需关
心模型结构。

**激活函数**：ReLU、Sigmoid（数值稳定分段实现）、Tanh、Softmax（逐行归一化，
含 log-sum-exp 稳定技巧）。每个激活函数同时提供导数实现（Sigmoid/Tanh 的导
数基于输出值计算，避免重复 exp）。

### 3.4 损失函数

- **MSELoss**：`(1/N) * Σ(pred - target)²`，用于回归。
- **CrossEntropyLoss**：数值稳定 log-sum-exp + 对数裁剪（epsilon 1e-12），支
  持向量和矩阵（逐行平均）输入，校验目标概率分布和为 1。

### 3.5 优化器

| 优化器 | 核心公式 | 状态 |
|--------|---------|------|
| SGD | `θ -= lr * g` | 无状态 |
| Momentum | `v = β*v + g; θ -= lr*v` | 速度向量 |
| Adam | `m=β₁*m+(1-β₁)g; v=β₂*v+(1-β₂)g²` + 偏差校正 | 一二阶矩估计 |

三个优化器共享 `Optimizer` trait。参数和梯度形状不匹配时拒绝更新且不修改参数
（panic-safety）。Adam 的 timestep 和速度向量会在参数形状变化时自动重置。

### 3.6 数据集

**合成数据集**：

- `linear_regression(samples, slope, intercept)` — 一维等距点，y = slope*x+intercept。
- `xor()` — 经典 4 行 XOR 真值表。
- `spiral(samples_per_class, classes)` — 多分类螺旋，one-hot 目标。坐标公式确
  定性生成，同参数 → 同数据。

**数据工具**（0.2.0 新增）：

- `Dataset::from_csv(name, text, num_features)` — 解析 CSV 文本，跳过表头和
  `#` 注释行，列数校验。
- `Dataset::iter_rows()` — `ExactSizeIterator`，按行惰性生成向量对。
- `Dataset::shuffle(seed)` — Fisher-Yates + LCG 确定性洗牌。
- `Dataset::split(ratio)` — 训练/测试集划分。

### 3.7 训练循环

**设计原则**：每个训练循环每 epoch 构建一次 `ComputationGraph`，损失梯度解析
计算为种子，参数梯度由 autograd 引擎自动计算。流程：

```
1. 构建图: Leaf(features, requires_grad=false)
          Leaf(weights, requires_grad=true)
          Leaf(bias, requires_grad=true)
2. 前向: MatMul → RowAdd → (Sigmoid/Softmax)
3. 种子: dL/dy = (2/N)*(pred-target) [MSE]
        dL/dz = (σ(z)-t)/N      [sigmoid+BCE]
        dL/dz = (s-t)/N          [softmax+CE]
4. backward_with_grad → 叶子节点梯度
5. take_gradients → optimizer.step
```

四个训练循环：
- `train_linear_regression` — MSE，单 Linear 层
- `train_binary_classification` — sigmoid+BCE，逻辑回归，含 accuracy
- `train_xor_mlp` — 2-2-1 sigmoid MLP，含隐层
- `train_spiral_classifier` — 极坐标特征映射 + Linear + softmax + CE

### 3.8 CLI 和报告

**命令**：

| 命令 | 默认 epochs | 可选参数 |
|------|------------|---------|
| `train-linear` | 120 | `--epochs --learning-rate --samples --slope --intercept --format --output --save-model` |
| `train-xor` | 160 | `--epochs --learning-rate --format --output --save-model` |
| `train-spiral` | 160 | `--epochs --learning-rate --samples-per-class --classes --format --output --save-model` |
| `inspect` | — | 运行三个示例并输出参数和预测摘要 |

`--format` 支持 `text`/`csv`/`markdown`。`--output DIR` 导出 `summary.md` +
`history.csv`。`--save-model PATH` 保存纯文本 checkpoint。

**输出示例**（XOR 5 epochs）：

```
XOR MLP training
epochs=5  initial_loss=0.242958  final_loss=0.232110
best_loss=0.232110  loss_improvement=0.010848  best_accuracy=1.000000
last=epoch=5 loss=0.232110 accuracy=1.000000
probabilities=[0.205530]; [0.791871]; [0.791871]; [0.206772]
classes=[0.000000]; [1.000000]; [1.000000]; [0.000000]
```

### 3.9 模型序列化

纯文本格式：

```text
2,2,2         ← rank,dim0,dim1
1.00000000000000000e0
2.00000000000000000e0
3.00000000000000000e0
4.00000000000000000e0
```

- 17 位有效数字保证 f64 回环精度。
- 多张量按参数顺序拼接。
- `save_linear`/`load_linear`、`save_xor_mlp`/`load_xor_mlp` 提供模型级回环。
- 自动创建目标路径父目录。

### 3.10 后端抽象

`Backend` trait 定义了 11 个核心运算的方法签名。`CpuBackend` 以零成本委托方
式实现——每个方法直接调用 `Tensor` 对应方法。Trait 是 object-safe 的。未来
GPU 后端只需实现同一 trait 即可接入。

---

## 四、质量保障

### 4.1 测试策略

| 层级 | 数量 | 覆盖内容 |
|------|------|---------|
| 库单元测试 | 270 | 每个模块的公开 API、边界条件、错误路径、收敛验证 |
| CLI 单元测试 | 13 | 参数解析、命令分发、输出格式化、错误消息、save-model |
| CLI 集成测试 | 9 | 编译后的二进制文件：退出码、stdout/stderr、CSV 格式、文件输出、help 内容 |
| **合计** | **292** | |

测试特色：
- 张量测试验证数学正确性（如 `matmul` 与单位矩阵保持值不变）。
- autograd 测试用标量手算值做精确对照（如 `z = x*y + x`，手算 dz/dx = y+1）。
- 训练测试验证 loss 下降和参数收敛（如线性回归 160 epochs 后 loss < 1e-6，
  XOR 120 epochs 后 accuracy = 1.0）。
- 确定性数据集保证测试结果跨机器一致。

### 4.2 CI/CD

GitHub Actions workflow（`.github/workflows/ci.yml`）在 push/PR 时自动运行：

```bash
cargo fmt --check
cargo build
cargo test
cargo clippy -- -D warnings
```

### 4.3 代码质量

- `cargo fmt` — 全项目 Rust 标准格式。
- `cargo clippy -- -D warnings` — 零警告。
- 所有公开 API 均有 `///` 文档注释 + `#[must_use]` 标记。
- Conventional commits 风格：`feat`/`test`/`refactor`/`docs`/`fix`/`style`。

---

## 五、项目进度

### 0.1.0 里程碑

Tensor、autograd（7 种梯度规则）、NN（Linear/Sequential/4 种激活）、Loss
（MSE/CE）、Optim（SGD/Momentum/Adam）、合成数据集、4 个训练循环（手工梯
度）、CLI（4 个命令）、报告导出、CI。232 测试。

### 0.2.0 里程碑

- autograd：补全 5 种梯度规则 + backward_with_grad + 数值验证
- 训练循环全部改用计算图反向传播，删除 ~190 行手工梯度
- 模型序列化（Linear/XorMlp）+ CLI --save-model
- 数据集扩展（CSV/shuffle/split/行迭代器）
- Backend trait + CpuBackend（GPU 架构预留）
- 292 测试

### 提交统计

共 18 个 commit（从初始到 0.2.0），净增约 4,000 行 Rust 代码 + 2,000 行文
档。零外部依赖，一键 `cargo test` 全绿。

---

## 六、核心设计决策回顾

| 决策 | 理由 |
|------|------|
| `f64` 而非泛型 `T: Float` | 降低类型复杂度，教学框架不需要多精度 |
| Shape 强制校验（拒绝零维、空数据） | 让错误尽早暴露，避免训练中途崩溃 |
| 梯度存图不存 Tensor | 分离关注点，Tensor 保持纯数据语义 |
| `GradientSet` 独立于计算图 | 优化器可脱离 autograd 单独测试和演示 |
| Synthetic 数据集 + 确定性初始化 | 跨平台可复现，适合评分 |
| 手工 CNN/RNN 缺失 | 项目定位为"框架基础"，非应用模型 zoo |
| 损失梯度手工计算作种子 | 保持教学可检查性，同时 autograd 仍承担参数梯度的传播 |
| 零外部依赖 | 降低编译复杂度，代码审查不涉及第三方库行为 |
| CLi 手写解析 | 标准库够用，减少认知负担 |

---

## 七、已知限制

1. **GPU 后端仅有 trait 接口**：无实际 GPU kernel。未来需要改造 `Tensor<B: Backend>`。
2. **序列化未覆盖 Sequential 泛化容器**：`--load-model` 也未实现。
3. **全批量梯度下降**：无 mini-batch 迭代，不适用于大数据集。
4. **ComputationGraph 每 epoch 重建**：无跨 epoch 复用，大图开销显著。
5. **Softmax 在图外计算种子梯度**：未利用图内 Softmax 反向路径（利用了
   softmax+CE 的简化公式）。
6. **无数据预处理**：归一化、标准化、特征工程需外部完成。
7. **仅支持 rank ≤ 2 张量**：CNN 所需的 4D 张量（NCHW）不支持。
8. **无分布式训练**：单进程单线程执行。

---

## 八、总结

RustGrad 成功实现了一个从张量运算到模型训练再到报告导出的完整深度学习流水
线，在零外部依赖、292 个测试全部通过的前提下，覆盖了：

- 完整的张量运算体系（15+ 种操作）
- 12 种梯度规则的反向模式自动求导引擎
- 带有限差分数值验证的梯度正确性保障
- 4 个训练示例（线性回归 → 二分类 → XOR MLP → 多分类螺旋）
- 3 种优化器（SGD / Momentum / Adam）
- 2 种损失函数（MSE / CrossEntropy）
- 模型序列化与数据加载基础
- CLI 工具链与报告导出
- GPU 后端架构预留

项目代码可作为 Rust 系统编程 + 深度学习基础的课程实验参考。
