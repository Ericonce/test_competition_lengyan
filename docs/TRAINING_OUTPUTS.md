# 每次训练的结果输出

所有模型共用同一训练输出流程，文件保存到 `model.train(output_dir=...)` 指定目录。无需修改各模型训练入口即可生效。

```text
results/<方法>/
├── best_model.pt                       # 有验证集：验证 Macro F1 最佳权重
├── last_model.pt                       # 最后完成轮次，可续训
├── final_model.pt                      # 无验证集：最终权重
├── training_history.csv                # 每轮指标，完成一轮就写入
├── final_metrics.json                  # 最佳或最终权重的最终评估
├── training.log                        # 参数、训练日志与结果；续训时追加
├── dataset_selection.json              # 采样、划分与序列路径（采用 pipeline 划分时）
├── normalization.npz                   # 训练数据标准化统计量
├── normalization.json                  # 完整归一化方案、偏移与缩放参数
├── resolved_pipeline.json              # 完整生效配置与实际运行信息
├── configs/
│   ├── pipeline.yaml                   # 完整生效配置，包含 train/loss/runtime
│   ├── data.yaml                       # 生效的数据配置
│   ├── augment.yaml                    # 生效的增强开关、顺序和阈值
│   └── model.yaml                      # 生效的网络配置
├── figures/
│   ├── accuracy_curve.png              # 训练、验证准确率
│   ├── macro_f1_curve.png               # 训练、验证 Macro F1
│   ├── training_curves.png              # 两张曲线合并
│   ├── val_per_class_metrics.png        # 验证集每类准确率与 F1
│   ├── val_confusion_matrix.png         # 按真实类别行归一化
│   └── val_tsne.png                     # 分类前的模型特征
├── val_classification_report.csv
├── val_classification_report.json
├── val_confusion_matrix.csv             # 原始计数
├── val_confusion_matrix_normalized.csv   # 行归一化数值
├── val_predictions.csv                  # 每条序列的标签、预测与置信度
└── tsne/
    ├── val_embeddings.npz               # 采样特征、样本索引、标签、预测、路径
    ├── val_tsne.csv                     # 二维坐标与对应标签、预测、路径
    └── settings.json                    # 实际采样及降维设置
```

默认开启 `visualize=True`，每张图同时输出 PNG、SVG、PDF，PNG 为 300 dpi。以上目录表示**有验证集且开启可视化**的情况。

在 `model.train(..., visualize=False)` 中关闭后，不生成 figures/ 内的图或 tsne/ 内的特征、坐标及降维设置，也不进行 t-SNE 计算。训练历史、指标、分类报告、混淆矩阵数值、预测、配置、日志和权重仍正常保存；本输出流程管理的旧图和旧 t-SNE 文件会清理。此开关控制结果文件绘制，不弹出桌面窗口。

## 指标及评估对象

`training_history.csv` 保留全部已有损失、学习率及耗时字段，并包含 `epoch`、`train_accuracy`、`val_accuracy`、`train_macro_f1`、`val_macro_f1`。训练指标是当轮训练批次的记录，可能受到训练增强、Dropout、类别均衡采样和不断更新的权重影响；验证指标在评估模式、不施加增强时计算。曲线代表一次训练的逐轮变化，不代表跨随机种子的均值或误差范围。

有验证集时，训练结束加载验证 Macro F1 最佳的 `best_model.pt`，在完整验证集上生成最终指标、逐序列预测、每类报告和混淆矩阵。t-SNE 仅使用验证集的模型 `stream_embedding`，即分类前的序列特征。

每类 `class_accuracy` = 该类预测正确数量 / 该类真实样本数量，与 Recall 相同；同时保存 Precision、Recall、F1、support 和 correct_count。没有真实样本的类别，class_accuracy 留空（JSON 为 null），F1 沿用项目的零除处理规则。混淆矩阵行是真实类别、列是预测类别；无真实样本的行归一化为 0。类别多于 20 时不打印每个格子的数值，避免遮挡，完整数值在 CSV。

Macro F1 沿用项目的 `macro_f1` 指标口径；配置全部类别的指标仍单独保存为 `all_class_macro_f1`。

## 没有验证集

当 `dataset.train_ratio=1.0`、`dataset.val_ratio=0.0`：

- 保存 `last_model.pt`、`final_model.pt`；不产生 best_model.pt。
- 保存训练历史及训练曲线；`val_accuracy`、`val_macro_f1` 留空，不绘制验证曲线。
- 最终权重对训练集无增强、按顺序逐条评估一次，避免均衡采样重复抽取导致统计偏差。
- 分类报告、计数/归一化矩阵、预测 CSV 以 `train_` 命名，仅保存数值文件。
- 跳过验证集每类指标图、混淆矩阵图和 t-SNE。
- 清理同一输出目录中的旧分类分析文件和图，避免混入此前有验证集的结果；只清理本流程管理的固定文件。

## 绘图与特征降维可视化（t-SNE）设置

每个 pipeline 的 `train.reporting` 提供默认值：

```yaml
train:
  visualize: true  # false 关闭所有绘图和 t-SNE，保留数值结果。
  reporting:
    dpi: 300
    formats: [png, svg, pdf]
    tsne:
      max_samples: 3000  # 按类别平衡采样；0 使用全部验证集
      seed: 2026
      perplexity: 30.0
      max_iter: 1000
```

也可以直接在训练调用中覆盖，代码优先：

```python
model.train(
    pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
    data=PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
    output_dir=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap",
    visualize=True,
    reporting={
        "formats": ["png"],
        "tsne": {"max_samples": 2000, "perplexity": 20.0},
    },
)
```

开启可视化时，PNG 始终输出。t-SNE 采样上限应至少覆盖验证集中实际存在的类别数量；采样确定后，仅收集这些样本的特征，预测和分类指标仍覆盖全部验证序列。perplexity 会调整到小于实际样本数；少于 3 条验证序列时保留特征文件并在 settings.json 写明跳过原因，不生成 t-SNE 图或坐标文件。

特征维度较大时先通过 PCA 降至最多 50 维，再运行二维 t-SNE。图例按真实类别着色，类别较多时结合形状区分；t-SNE 用于描述特征分布，不能单凭聚类外观认定模型分类效果或模块贡献。图计算在 CPU 上进行，适用于 CUDA、昇腾 NPU 和 CPU 训练后的输出。

## 配置与续训

训练开始打印合并代码覆盖后的配置，保存到 configs/ 和 training.log。这些文件是实际生效配置的快照，不是原始 YAML 的简单拷贝。runtime 记录实际设备、AMP 状态、续训起点及优化器初始学习率/权重衰减；续训恢复优化器时，实际学习率可能不同于本次传入的 lr。

续训沿用已有 history，曲线使用累计 epoch 编号。training.log 追加本次配置及日志，新的配置快照反映本次调用。同一输出目录重新从头训练时覆盖当前日志及结果，建议不同实验使用不同 output_dir。

后处理有进度提示，包括最终预测采集、曲线输出、验证集分析和 t-SNE。PNG/PDF/SVG 导出及 t-SNE 显示阶段耗时，每秒刷新；`data_cfg={"progress": False}` 可以关闭提示。
