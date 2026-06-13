# PPT 第 8 页 & 第 10 页 —— 演示输出与讲解指南

本目录包含答辩 PPT 第 8 页（训练循环）和第 10 页（CLI 与报告）所需的命令
输出。

---

## 第八页：训练循环（四个示例的递进演示）

### 8a. 线性回归 —— 验证参数收敛

**命令**：

```bash
rustgrad train-linear --epochs 120 --learning-rate 0.12 --samples 31
```

**实际输出**：

```text
Linear regression training
epochs=120
initial_loss=0.994192
final_loss=0.000000
best_loss=0.000000
loss_improvement=0.994192
last=epoch=120 loss=0.000000
weight=1.499966
bias=0.750000
```

**讲解要点**：

> 第一个示例是最简单的线性回归。数据集是 y = 1.5x + 0.75 的 31 个等距点。
> 训练 120 轮后 loss 从 0.99 降到几乎为零。最后一行打印了学到的参数——weight
> ≈ 1.5、bias ≈ 0.75，与生成数据的真实参数一致。这说明我们的梯度计算、优化
> 器更新、训练循环都是正确的。

---

### 8b. 二分类 —— 逻辑回归 + accuracy

**讲解要点（无独立运行截图，可在 PPT 上口头带过）**：

> 二分类在单层 Linear 后面加了 sigmoid 激活，用二元交叉熵做损失。目标必须是
> 0 或 1。和线性回归的区别在于多了一个 accuracy 指标——预测值大于阈值 0.5 判
> 为 1，否则为 0。

---

### 8c. XOR MLP —— 非线性决策边界

**命令**：

```bash
rustgrad train-xor --epochs 160 --learning-rate 0.4
```

**实际输出**：

```text
XOR MLP training
epochs=160
initial_loss=0.242958
final_loss=0.150075
best_loss=0.150075
loss_improvement=0.092884
best_accuracy=1.000000
last=epoch=160 loss=0.150075 accuracy=1.000000
probabilities=[0.155609]; [0.879026]; [0.879026]; [0.159095]
classes=[0.000000]; [1.000000]; [1.000000]; [0.000000]
```

**讲解要点**：

> XOR 是一个经典的非线性可分问题——单层线性模型不可能正确分类。这里我们用了
> 一个 2-2-1 的小型 sigmoid MLP。注意看最后的 classes 输出：四行恰好是
> [0,1,1,0]，这刚好就是 XOR 真值表。accuracy 达到了 100%。
>
> 这个示例的价值在于展示了非线性激活函数的必要性——隐层的 sigmoid 把二维输
> 入映射到了一个线性可分的空间。

**关键数据解读**：
- XOR 真值表: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
- 模型的 classes 输出: `[0] [1] [1] [0]` —— 完全匹配

---

### 8d. 螺旋多分类 —— 特征映射 + Softmax

**命令**：

```bash
rustgrad train-spiral --epochs 160 --learning-rate 0.7 --samples-per-class 12 --classes 3
```

**实际输出**：

```text
Spiral softmax training
epochs=160
initial_loss=1.386677
final_loss=0.050991
best_loss=0.050991
loss_improvement=1.335687
best_accuracy=0.972222
last=epoch=160 loss=0.050991 accuracy=0.972222
classes=3
samples_per_class=12
weight_shape=[2, 3]
```

**讲解要点**：

> 螺旋是最复杂的一个示例。原始 (x, y) 坐标画出来是三条互相缠绕的螺旋线，线
> 性分类器完全不可能分开。我们的做法是做一个极坐标特征映射——用半径和角度来
> 重新表达每个点——然后接一个线性 softmax 分类头。
>
> 注意这个数字：loss 从 1.39 降到了 0.05，accuracy 达到了 97.2%。整个分类器
> 只有一个 Linear 层（2×3 权重矩阵），但因为前面的特征映射是非线性的，它足
> 够把螺旋分开。这很好地展示了一个机器学习思想：合适的特征变换可以大幅降低
> 分类难度。

---

### 8e. PPT 排版建议

建议在 PPT 上并排展示四个示例的输出，形成对比表格：

| 示例 | 模型 | epochs | 初始 loss | 最终 loss | 最佳 accuracy |
|------|------|--------|----------|----------|-------------|
| 线性回归 | 1×1 Linear | 120 | 0.994 | 0.000 | — |
| XOR MLP | 2-2-1 Sigmoid | 160 | 0.243 | 0.150 | 100% |
| Spiral | 2→2→3 Softmax | 160 | 1.387 | 0.051 | 97.2% |

---

## 第十页：CLI 与报告导出（三种输出模式）

### 10a. text 格式 —— 人类可读摘要（默认）

参见第 8 页各命令输出，`--format text` 输出含摘要统计和预测详情。

### 10b. CSV 格式 —— 数据处理

**命令**：

```bash
rustgrad train-xor --epochs 5 --format csv
```

**实际输出**：

```csv
epoch,loss,accuracy
1,0.242958,1.000000
2,0.239091,1.000000
3,0.236178,1.000000
4,0.233919,1.000000
5,0.232110,1.000000
```

**讲解要点**：

> `--format csv` 输出的每一行是一个 epoch 的 loss 和 accuracy。可以直接导入
> Excel 或 Python pandas 画 loss 曲线。

### 10c. Markdown 格式 —— 报告就绪

**命令**：

```bash
rustgrad train-xor --epochs 5 --format markdown
```

**实际输出**：

```markdown
# XOR MLP training

## Summary
- Epochs: 5
- Initial loss: 0.242958
- Final loss: 0.232110
- Best loss: 0.232110
- Loss improvement: 0.010848
- Best accuracy: 1.000000

## History
| epoch | loss | accuracy |
| ---: | ---: | ---: |
| 1 | 0.242958 | 1.000000 |
| ... | ... | ... |
```

**讲解要点**：

> Markdown 格式可以直接粘到 GitHub Issue 或实验报告里，标题、摘要、历史表格
> 一应俱全。

### 10d. 报告文件导出

**命令**：

```bash
rustgrad train-spiral --epochs 5 --samples-per-class 4 --classes 3 \
  --output /tmp/demo --save-model /tmp/demo/spiral.checkpoint
```

**实际输出**：

```text
Spiral softmax training
epochs=5
initial_loss=1.383407
final_loss=0.668863
...
best_accuracy=0.916667

report_dir=/tmp/demo
markdown=/tmp/demo/summary.md
csv=/tmp/demo/history.csv

classes=3
samples_per_class=4
weight_shape=[2, 3]
model_saved=/tmp/demo/spiral.checkpoint
```

**讲解要点**：

> `--output DIR` 会在目标目录下写入两个文件：`summary.md`（Markdown 报告）和
> `history.csv`（CSV 数据）。`--save-model PATH` 会额外保存模型参数。训练结束
> 后 CLI 会告诉你每个文件的确切路径。
>
> 这三个选项——format 控制终端输出、output 导出报告文件、save-model 保存权重
> ——覆盖了从快速检查到正式实验记录的完整需求。

### 10e. inspect 命令 —— 快速检查所有模型

**命令**：

```bash
rustgrad inspect
```

**实际输出**：

```text
RustGrad model inspection
linear.weight=1.499976
linear.bias=0.750000
xor.probabilities=[0.155609]; [0.879026]; [0.879026]; [0.159095]
xor.classes=[0.000000]; [1.000000]; [1.000000]; [0.000000]
spiral.weight_shape=[2, 3]
spiral.best_accuracy=0.958333
spiral.preview_probabilities=[0.983607, 0.007818, 0.008575]; ...
```

**讲解要点**：

> `inspect` 命令无需参数，一键运行三个训练示例并打印参数和预测摘要。适合答
> 辩现场做快速演示——一个命令看到全部核心功能。

---

## PPT 第 10 页排版建议

```
┌────────────────────────────────────────────────┐
│  CLI 四种输出模式                               │
│                                                │
│  📊 text   命令行摘要 + 参数 + 预测             │
│  📈 csv    可导入 Excel / Python 画 loss 曲线  │
│  📝 md     GitHub / 实验报告直接粘贴            │
│  💾 --output DIR  导出 summary.md + history.csv │
│  💾 --save-model PATH  导出 checkpoint          │
│                                                │
│  🔍 rustgrad inspect  一键演示所有模型          │
└────────────────────────────────────────────────┘
```
