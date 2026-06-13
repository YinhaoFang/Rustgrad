# RustGrad PPT 图表示例

本目录包含 8 张 PlantUML 图，对应答辩 PPT 各页。

## 使用方法

### 在线渲染（最快）

1. 打开 https://www.plantuml.com/plantuml/uml/
2. 复制 `.puml` 文件内容粘贴进去
3. 点击 Submit，下载 PNG/SVG

### VS Code 插件

安装 `jebbs.plantuml` 扩展，在 `.puml` 文件内 `Alt+D` 预览，右键导出图片。

### 命令行

```bash
# 需要安装 Java 和 plantuml.jar
java -jar plantuml.jar diagrams/*.puml
```

## 图表清单

| 文件 | 类型 | 对应 PPT | 内容 |
|------|------|---------|------|
| `01_architecture.puml` | 包图 | 第 3 页 | RustGrad 分层架构：界面→编排→学习→求导→张量→数据 |
| `02_tensor.puml` | 类图 | 第 4 页 | Tensor + Shape 完整 API 和关系 |
| `03_autograd_core.puml` | 类图 | 第 5 页 | ComputationGraph / GraphNode / NodeId / Operation |
| `04_backward_flow.puml` | 序列图 | 第 5 页 | 反向传播五步流程：种子→拓扑→逆序→梯度→累积 |
| `05_nn_module.puml` | 类图 | 第 6 页 | Module trait / Linear / Sequential / Activation |
| `06_optimizer.puml` | 类图 | 第 7 页 | Optimizer trait / SGD / Momentum / Adam / GradientSet |
| `07_training_loop.puml` | 序列图 | 第 8 页 | 单 epoch 全流程（以 XOR MLP 为例） |
| `08_data_pipeline.puml` | 类图 | 第 9 页 | Dataset 完整 API / 序列化 / Backend trait |

## PPT 嵌入建议

- **类图**（02/03/05/06/08）：适合独立成页，字号调整到演讲厅最后一排能看清
- **序列图**（04/07）：适合逐步骤动画展示——先遮住后续步骤，配合讲稿逐步揭示
- **包图**（01）：适合作为目录页或总结页的视觉锚点
