# RustGrad 0.2.0 开发报告

## 概述

RustGrad 是一个用 Rust 实现的面向教学的深度学习框架。0.1.0 版本建立了基础
设施：张量运算、自动求导、神经网络层、损失函数、优化器、合成数据集和训练循
环。0.2.0 的目标是解决 0.1.0 中记录的四个已知限制：

1. 自动求导只覆盖示例所需的部分操作
2. 没有模型 checkpoint 序列化格式
3. 没有大型数据集加载器
4. 没有 GPU 后端

每个限制都在不引入外部依赖的前提下得到了处理。以下按模块汇报工作内容。

---

## 1. 自动求导补全

### 问题

0.1.0 的 `local_gradients` 方法对 `Transpose`、`ReLU`、`Sigmoid`、`Tanh`、
`Softmax` 五种操作返回 `UnsupportedOperation` 错误。同时，四个训练循环全部使
用手工推导的梯度公式，`ComputationGraph::backward()` 从未在训练代码中使用。

### 完成的工作

**梯度规则实现**（`src/autograd/mod.rs`）：

| 操作 | 梯度公式 | 实现要点 |
|------|---------|---------|
| `Transpose` | `grad_input = upstream.transpose()` | 单操作数梯度，直接转置 |
| `ReLU` | `grad = upstream * relu_derivative(input)` | 输入 >0 时导数为 1，否则 0 |
| `Sigmoid` | `grad = upstream * σ(output) * (1 - σ(output))` | 使用输出值计算导数，避免重复求 exp |
| `Tanh` | `grad = upstream * (1 - tanh²(output))` | 使用输出值计算导数 |
| `Softmax` | `grad[i] = s[i] * (g[i] - Σ_j s[j] * g[j])` | 支持向量和逐行矩阵，雅可比通过向量-雅可比积实现 |

**新的图操作方法**：

- `ComputationGraph::backward_with_grad(output, seed)` — 支持用自定义梯度种子
  初始化反向传播，这是训练循环接入 autograd 引擎的关键。
- `ComputationGraph::take_gradients()` — 从叶子节点取出累积梯度，直接供优化
  器使用。

**新增 Tensor 操作**：

- `Tensor::row_add(bias)` — 将向量广播叠加到矩阵每一行，用于 Linear 层的
  偏置加法。
- `Operation::RowAdd` — 对应的计算图操作，梯度规则为矩阵梯度原样传递、偏置
  梯度按行求和。

**梯度正确性验证**：

所有五个新梯度规则都通过了双重量验证：
- **解析验证**：将 autograd 输出与手算公式对比。
- **有限差分验证**：`check_gradient_numeric` 框架对每个叶节点值加微小扰动
  `(f(x+h) - f(x-h)) / 2h`，与 autograd 梯度比较（容差 1e-5）。

---

## 2. 模型序列化

### 问题

0.1.0 没有保存/加载模型参数的机制。训练结果只能用 `--output DIR` 导出损失
曲线，无法复用已训练的权重。

### 完成的工作

**纯文本序列化格式**（`src/serialize.rs`）：

```text
rank,dim0,dim1,...
value0
value1
...
```

- 第一行为 `rank,dim0,dim1,...`，标明张量形状。
- 后续每行一个 `f64` 值，17 位有效数字保证回环精度。
- 多张量按参数顺序拼接。
- 无需 JSON/二进制依赖，人类可直接阅读。

**模型级 API**：

- `save_linear(linear, path)` / `load_linear(path)` — Linear 层回环。
- `save_xor_mlp(model, path)` / `load_xor_mlp(path)` — XOR 两层 MLP 回环。
- `write_tensor(tensor, &mut String)` / `read_tensor(&str)` — 通用张量序列化。

**CLI 集成**（`src/main.rs`）：

三个训练命令均支持 `--save-model PATH` 选项。训练结束后自动保存模型参数，
输出中显示 `model_saved=PATH`。自动创建目标路径的父目录。

---

## 3. 数据集扩展

### 问题

0.1.0 只有三个硬编码的合成数据集，无法从外部文件加载数据，也没有训练/
验证集划分和打乱功能。

### 完成的工作

**CSV 加载**（`Dataset::from_csv`）：

- 第一行自动作为表头跳过。
- 支持 `#` 开头的注释行和空行。
- `num_features` 参数指定前几列为特征，剩余列为目标。
- 每行校验浮点数合法性和列数。

**行迭代器**（`Dataset::iter_rows`）：

- 实现 `ExactSizeIterator`，每次返回 `(features_vec, targets_vec)` 对。
- 按行按需生成向量，不额外分配全量内存。

**确定性洗牌**（`Dataset::shuffle(seed)`）：

- Fisher-Yates 洗牌算法。
- 使用 64 位线性同余生成器（LCG），`seed` 相同则排列相同。
- 单行数据集直接返回原数据。

**训练/测试集划分**（`Dataset::split(ratio)`）：

- `ratio` 为训练集比例（如 0.8 表示 80/20 划分）。
- 保留子集，分别命名为 `{name}_train` 和 `{name}_test`。

---

## 4. 后端抽象

### 问题

0.1.0 的 Tensor 运算直接实现在 `Tensor` 上，没有为 GPU 后端预留接口。

### 完成的工作

**`Backend` trait**（`src/backend.rs`）：

定义了覆盖全部核心运算的 trait 方法：

```rust
pub trait Backend: std::fmt::Debug {
    fn name(&self) -> &str;
    fn add(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;
    fn sub(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;
    fn mul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;
    fn div(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;
    fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;
    fn transpose(&self, tensor: &Tensor) -> Result<Tensor>;
    fn sum(&self, tensor: &Tensor) -> Result<Tensor>;
    fn mean(&self, tensor: &Tensor) -> Result<Tensor>;
    fn sum_axis(&self, tensor: &Tensor, axis: usize) -> Result<Tensor>;
    fn mean_axis(&self, tensor: &Tensor, axis: usize) -> Result<Tensor>;
    fn row_add(&self, matrix: &Tensor, bias: &Tensor) -> Result<Tensor>;
}
```

**`CpuBackend`**：

- 零成本委托实现：每个方法直接调用 `Tensor` 对应方法。
- trait 是 object-safe 的（已验证 `&dyn Backend` 可用）。
- 未来 GPU 后端可以直接实现同一 trait，不改变调用方 API。

---

## 5. 测试覆盖

### 新增测试统计

| 测试类别 | 数量 | 说明 |
|---------|------|------|
| autograd 单元测试 | 39 | 含 5 个解析验证 + 5 个有限差分数值验证 + 29 个存量测试 |
| serialize 单元测试 | 9 | Tensor/Linear/XorMlp 回环、错误路径、前向输出一致性 |
| data 单元测试 | 26 | CSV 解析、行迭代、洗牌、划分，含 13 个新测试 |
| backend 单元测试 | 4 | CpuBackend 正确性、trait object 可用性 |
| CLI 单元测试 | 13 | 含 2 个 `--save-model` 测试 |
| CLI 集成测试 | 9 | 含 3 个新测试：save/load 回环、help 输出 |
| **总计** | **292** | |

### 质量检查

```bash
cargo fmt --check   # 零差异
cargo test          # 292 passed, 0 failed
cargo clippy -- -D warnings   # 零警告
```

---

## 6. 架构变化

### 新增文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/serialize.rs` | ~310 | 模型 checkpoint 序列化/反序列化 |
| `src/backend.rs` | ~180 | Backend trait + CpuBackend |

### 修改文件

| 文件 | 变化 |
|------|------|
| `src/autograd/mod.rs` | +600 行：5 个梯度规则、backward_with_grad、take_gradients、有限差分框架、11 个测试 |
| `src/train/mod.rs` | +70/-190 行：四个训练循环用计算图重写，删除手工梯度代码 |
| `src/tensor/mod.rs` | +25 行：row_add 操作 |
| `src/data/mod.rs` | +320 行：CSV 加载、行迭代器、shuffle、split |
| `src/main.rs` | +150 行：--save-model 解析和保存逻辑、2 个新测试 |
| `src/lib.rs` | +2 行：注册 backend 和 serialize 模块 |

---

## 7. 0.2.0 已知限制

1. **GPU 后端仅有接口**：`Backend` trait 和 `CpuBackend` 是架构预留。未来需
   要在 `Tensor` 上引入泛型参数 `B: Backend`，并在 `GpuBackend` 中实现真正的
   GPU kernel（wgpu 或 CUDA 绑定），当前不能在 GPU 上运算。

2. **序列化仅覆盖 Linear 和 XorMlp**：`Sequential` 容器的泛化序列化（遍历
   所有 layer 的参数）尚未实现。训练前从文件恢复权重的 `--load-model` CLI
   选项也未实现。

3. **数据集加载器无外部文件系统集成**：`from_csv` 接受字符串内容，未提供从
   文件路径自动读取的 convenience 方法。数据预处理（归一化、标准化）功能未
   实现。

4. **`ComputationGraph` 每 epoch 重建**：训练循环在每轮从零构建计算图，放弃
   了跨 epoch 的图复用。对于当前的小规模示例数据集影响可忽略，但大规模训练
   时图构建开销会成为瓶颈。

5. **Softmax 在训练图中不参与反向传播**：`train_spiral_classifier` 在计算图
   外调用 `softmax` 获取概率后手工计算 `(softmax(logits) - targets)/N` 作为
   种子梯度，图内的 Softmax 节点未接入反向路径。这是因为标准的
   softmax+CE 组合梯度公式可以简化为直接减法，当前实现利用了这一点以保持训
   练循环简洁。

6. **无批量训练**：所有训练例子都在全量数据集上做梯度下降（full-batch
   gradient descent），batch size 等于数据集大小。虽然 `Dataset::batch` 和
   数据迭代器已支持按批采样，但训练循环尚未实现 mini-batch 迭代。

---

## 8. 提交记录

| Commit | 内容 |
|--------|------|
| `4ec66e3` | feat(autograd): 补全 Transpose/ReLU/Sigmoid/Tanh/Softmax 梯度规则 |
| `57f1f0d` | test(autograd): 11 个单元测试 + 有限差分数值验证框架 |
| `cf92b2e` | feat(train): 四个训练循环全部改用 ComputationGraph::backward_with_grad |
| `5706013` | refactor(train): 删除 191 行手工梯度死代码 |
| `d481b71` | feat(serialize): 纯文本模型 checkpoint 序列化 |
| `9df2876` | feat(cli): --save-model PATH CLI 选项 |
| `7581dc4` | feat(data): CSV 加载器、行迭代器、shuffle、train/test split |
| `3844ccc` | feat(backend): Backend trait + CpuBackend |
| `d4c6c88` | test(cli): save-model 回环集成测试 |
| `d5a6b4d` | style: cargo fmt 全项目格式化 |
