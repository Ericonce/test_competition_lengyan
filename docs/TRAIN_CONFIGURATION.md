# 训练与推理参数设置

新数据集的目录格式、data.yaml 字段和完整操作流程见 [新数据集接入指南](NEW_DATASET_GUIDE.md)。修改数据路径时同步调整类别数、采样比例和结果目录。

全部 train/infer 入口保留 PROJECT_ROOT 用于跨机器定位项目，其他路径直接填写在 model.train()/model.infer() 内，不再定义 PIPELINE_YAML、DATA_YAML、OUTPUT_DIR、CHECKPOINT 等路径常量。

常用参数已经列在调用中，可直接修改。普通训练入口保持原 YAML 的轮数、批量大小和采样比例；train_small.py 保留小数据集路径、10 类和 50 轮。MambaSL 保持关闭 AMP。类别数需要与训练数据的标签范围匹配。

优先级：训练类默认值 < YAML < model.train()/model.infer() 显式参数。已填写的数值优先于 YAML；需要让 YAML 控制某项时，删除对应参数。num_classes=None 表示继承 pipeline，再按数据或权重自动读取。未显式填写的网络结构、损失等仍继承 YAML。

## 训练入口

```python
return model.train(
    pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
    data=PROJECT_ROOT / "datasets/synthetic_radar_overlap_small/data.yaml",
    output_dir=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap_small",
    num_classes=10,
    epochs=50,
    batch_size=64,
    lr=0.001,
    device="auto",
    resume=False,
    resume_checkpoint=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap_small/last_model.pt",
    sequence_length=256,
    length_policy={"short": "zero", "long": "head"},
    dataset={"sample_ratio": 0.01, "train_ratio": 0.8, "val_ratio": 0.2},
    augment={"preset": PROJECT_ROOT / "configs/augment/default.yaml", "enabled": False},
)
```

完整入口还列出优化器、权重衰减、早停、AMP、工作进程数、种子、缓存和数据加载进度。续训时设置 resume=True，填写断点路径；epochs 是本次额外训练轮数。全部数据训练可设置 sample_ratio=1.0、train_ratio=1.0、val_ratio=0.0。

早停支持 `patience=10` 简写，与 `early_stopping_patience=10` 等价：连续 10 轮验证 Macro-F1 未改善时停止。也支持 `train={"patience": 10}`、`**{"train.patience": 10}` 和 YAML 的 `train.patience`。建议只写其中一种；同一配置层两种名称都填写时，非 None 的 patience 优先。跨层仍遵循代码优先于 YAML、直接训练参数优先于 train 字典的规则。patience=None 不覆盖已有设置；无验证集时不启用基于验证的早停。

## 推理入口

```python
return model.infer(
    pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
    source=PROJECT_ROOT / "datasets/synthetic_radar_test_small",
    checkpoint=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap_small/best_model.pt",
    save_dir=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap_small/infer_test",
    labels_file="",  # 只预测；填写测试目录的 labels.txt 路径才评估。
    num_classes=10,
    device="auto",
    sequence_length=256,
    length_policy={"short": "zero", "long": "head"},
    inference={"batch_size": 256, "progress": True},
    augment={"preset": PROJECT_ROOT / "configs/augment/default.yaml"},
)
```

推理直接读取平铺 CSV 文件夹，不使用测试 data.yaml。save_dir 是 CSV 输出目录；checkpoint 是显式权重路径，无验证集训练后用 final_model.pt。推理不执行随机增强，增强 YAML 引用只用于配置。

infer/infer_vote.py 同样直接填写各模型的 pipeline、checkpoint 与 enabled，并列出统一的 num_classes、batch_size、长度和进度参数。通用 infer/infer.py 保留 CLI；--pipeline 指定其他模型时，未提供的选项继承其 YAML。投票 --config 同理。

## 修改详细配置

- configs/pipeline/*.yaml：未在调用中覆盖的训练、损失、数据采样和推理设置。
- configs/model/*.yaml：网络结构参数。
- 数据集内 data.yaml：文件根目录、辐射源类别数、CF/PW/PRI 特征等数据描述。
- configs/augment/default.yaml：各增强方法开关、触发概率、阈值和执行顺序。训练入口将 augment.enabled 改为 True 后启用。

所有配置字段仍可通过 train、model（或 model_cfg）、data_cfg、augment、loss、inference 字典覆盖；字典递归合并。model/data 接受替换 YAML 路径，augment.preset 接受增强 YAML 路径。也支持点分路径，例如 **{"augment.methods.jittering.std": 0.02, "model.dropout": 0.2}。

同一字段重复指定时，点分路径优先于字典，直接传入的 epochs/lr 等优先于 train 字典。直接参数的 None 忽略；字典或点分路径中的 None 可清空字段。未知参数会报错，避免拼写错误被忽略。

## 训练与推理类别数

各 synthetic pipeline 都提供两个独立设置：

```yaml
train:
  num_classes: null  # 自动读取训练 data.yaml；可手动填写 10、100 等正整数
inference:
  num_classes: null  # 自动读取权重；手动填写时以此输出类别数为准
```

例如小型数据集可将两项都设置为 `10`。训练配置决定分类层输出数量，并记录到权重中；文件夹推理默认读取权重中的类别数。手动设置推理类别数时，以 inference.num_classes 为准；权重仍完整加载。数量减少时保留标签 0 到 N-1 的分类分数，再重新归一化概率；数量增加时扩展输出空间，新增未训练类别概率为 0。新增类别要具有识别能力，需要训练对应权重。终端打印推理类别数与原权重类别数，inference_summary.json 记录实际 num_classes。

代码优先于 YAML：`model.train(..., num_classes=10)` 和 `model.infer(..., num_classes=10)` 分别覆盖对应类别数，也支持 `train={"num_classes": 10}`、`inference={"num_classes": 10}` 或 dotted keys。命令行 train/predict/val 支持 `--num-classes 10`。

类别数不会筛选或重编号数据。所有标签须处于 `0..num_classes-1`，加载 CSV 和缓存前都会检查；要训练较少类别，应指定匹配的小型数据集。pipeline 默认 null 时可自动读取数据/权重；现有入口显式填写的 100 或 10 优先，因此接入新数据集时也需修改调用中的类别数。

