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
    labelsmooth=0.0,  # 0 关闭标签平滑；例如 0.1 开启。
    visualize=True,  # 默认绘图；False 跳过所有训练结果图和 t-SNE。
    normalization="zscore",  # zscore / minmax / maxabs / robust / log_zscore / none。
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

所有模型共用训练进度显示：每个 epoch 分别显示训练和验证的 batch 完成比例、耗时、预计剩余时间、累计平均 loss 与准确率，并显示当前处于加载数据、传输数据、前向计算、反向传播或更新权重哪个阶段。耗时每秒刷新，batch 完成后才增加进度；首个 batch 较慢时也能看到当前阶段。沿用 `data_cfg={"progress": True}` 开关，默认开启；设为 False 关闭训练和数据加载进度。

`progress_metric_interval=10` 控制 loss/acc 刷新间隔；进度完成数仍逐 batch 更新，最后一个 batch 一定刷新指标。损失与预测先留在设备上汇总，避免每个 batch 多次把标量和标签复制到 CPU。设为 `1` 时每批刷新指标，但增加设备同步。

训练可用 `use_amp=True, amp_dtype="float16"` 指定混合精度，也支持 `"bfloat16"` 和 `"auto"`。默认保持 FP16；BF16 不使用梯度缩放。推理需要在 `inference={"use_amp": True, "amp_dtype": "float16"}` 中单独设置；Transformer、TCN 的 pipeline 和推理入口已开启，其他模型默认保持原来的推理精度。混合精度可能造成少量数值差异。

Transformer 在 NPU 上默认通过原有权重执行显式编码器，避开 `aten::_transformer_encoder_layer_fwd` 的 CPU 回退；半精度优先使用 `torch_npu.npu_fusion_attention`，关闭 AMP 时使用 NPU 矩阵注意力。模型 YAML 的 `npu_attention_backend` 支持 `auto`（默认）、`fusion`、`math`；`native` 仅用于原实现性能对照，在旧版 torch_npu 上可能回退 CPU。CUDA/CPU 沿用 PyTorch 编码器，权重键名和形状保持兼容。

TCN 默认 `npu_conv_backend: native`，Ascend910B3 上原生卷积实测比 `phase`、`linear` 两种等价替代实现更快。更换环境时可用 `tools/benchmark_npu_sequences.py` 测试，不能仅根据算子名称推断速度。

所有训练入口都提供 `labelsmooth=0.0`，用于分类交叉熵的标签平滑：`0` 关闭，`0.1` 表示平滑强度为 0.1，有效范围为 `[0, 1]`。也可修改 pipeline 的 `loss.ce.label_smoothing`；显式 `labelsmooth` 优先于 YAML 和 `loss` 字典或点分路径。删除该参数或设为 `None` 时继承配置。此参数适用于所有模型的分类损失，不改变对比损失或评估标签。命令行训练也支持 `--labelsmooth 0.1`。

早停支持 `patience=10` 简写，与 `early_stopping_patience=10` 等价：连续 10 轮验证 Macro-F1 未改善时停止。也支持 `train={"patience": 10}`、`**{"train.patience": 10}` 和 YAML 的 `train.patience`。建议只写其中一种；同一配置层两种名称都填写时，非 None 的 patience 优先。跨层仍遵循代码优先于 YAML、直接训练参数优先于 train 字典的规则。patience=None 不覆盖已有设置；无验证集时不启用基于验证的早停。

## 推理入口

归一化可通过 `normalization="minmax"` 等设置，详细参数及各方案公式见[归一化设置](NORMALIZATION.md)。所有模型共用该配置；默认保持 Z-score。推理与续训自动使用权重保存的方案及训练统计。

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

文件夹推理、划分数据推理和多模型投票均显示实时进度，包含已完成序列数、当前 batch、耗时、预计剩余时间，以及读取、设备传输、预测或收集结果的阶段。使用 `inference={"progress": True}` 控制普通推理，投票入口使用 `progress=True`；默认均开启。耗时刷新表示进度显示仍在运行，完成数才表示实际完成的工作量。

infer/infer_vote.py 同样直接填写各模型的 pipeline、checkpoint 与 enabled，并列出统一的 num_classes、batch_size、长度和进度参数，无需命令行参数。更换投票 YAML 时直接修改 config 路径；显式参数仍优先于 YAML。通用 infer/infer.py 保留 CLI；--pipeline 指定其他模型时，未提供的选项继承其 YAML。

## 修改详细配置

所有方法训练结束自动输出曲线、分类报告和验证集 t-SNE，见 [训练结果输出](TRAINING_OUTPUTS.md)。绘图参数位于 pipeline 的 train.reporting，也可使用 model.train(reporting={...}) 覆盖。

所有训练入口都提供 `visualize=True`。设为 `False` 时跳过训练曲线、验证集每类指标图、混淆矩阵图以及 t-SNE 的特征采集、计算和导出；指标 CSV/JSON、混淆矩阵数值、预测 CSV、配置、日志和权重仍保存。也可设置 pipeline 的 `train.visualize`；直接参数优先，删除参数或设为 `None` 时继承 YAML。可视化是保存图片到结果目录，不弹出桌面窗口，适用于服务器训练。

- configs/pipeline/*.yaml：未在调用中覆盖的训练、损失、数据采样和推理设置。
- configs/model/*.yaml：网络结构参数。
- 数据集内 data.yaml：文件根目录、辐射源类别数、CF/PW/PRI 特征等数据描述。
- configs/augment/default.yaml：各增强方法开关、触发概率、阈值和执行顺序。训练入口将 augment.enabled 改为 True 后启用。

所有配置字段仍可通过 train、model（或 model_cfg）、data_cfg、augment、loss、inference 字典覆盖；字典递归合并。model/data 接受替换 YAML 路径，augment.preset 接受增强 YAML 路径。也支持点分路径，例如 **{"augment.methods.jittering.snr_db": 30.0, "model.dropout": 0.2}。

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

