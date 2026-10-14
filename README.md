# Synthetic Radar Pulse Classification

本项目按辐射源分类，输入为三通道 CF/PW/PRI。现有入口默认使用 synthetic 数据，符合相同 CSV 格式的新数据集也可通过独立 data.yaml、pipeline 和调用参数接入。模型结构与信号假设分支保留。

## 生成数据

```powershell
python .\tools\generate_synthetic_radar.py
python .\tools\generate_synthetic_radar_test.py
```

生成参数在 `configs/generator/synthetic_radar.yaml`。训练/验证先采样总池再划分，比例、固定输入长度与补长策略由 pipeline 提供默认值，训练调用中的显式参数优先。测试集平铺 CSV 和可选 labels.txt，不使用测试 data.yaml。

## 训练与推理

```powershell
python .\train\cnn_softmax_synthetic\train.py
python .\train\resnet_softmax_synthetic\train.py
python .\train\modular_multigrain_iq_synthetic\train.py
python .\infer\infer_cnn_softmax_synthetic.py
python .\infer\modular_multigrain_iq_synthetic\infer.py
python .\infer\resnet_softmax_synthetic\infer.py
```

19 种独立方法都有 `configs/pipeline/<方法>_synthetic.yaml`、`train/<方法>_synthetic/train.py` 和 `infer/<方法>_synthetic/infer.py`，另有 CNN synthetic baseline 配置。新增 ResNet1D-34/50、TCN、InceptionTime、LSTM/BiLSTM、Transformer Encoder、Informer Encoder 与 MLP；模型架构参考及分类适配见[经典序列模型说明](docs/CLASSIC_SEQUENCE_MODELS.md)。各方法保留自己的网络和损失定义，默认使用同一合成总池、采样与划分设置。

训练和推理入口的路径直接写在 model.train()/model.infer() 调用中，并列出类别数、批量大小、长度策略、增强 YAML 等常用设置；训练还列出轮数、学习率、采样/划分和续训，推理显式列出权重路径。代码参数优先于 YAML；删除对应参数可恢复继承 YAML。设备默认优先 CUDA，再 NPU，最后 CPU。数据加载、推理与缓存操作自动显示终端进度。

推理范例 `infer/infer_cnn_softmax_synthetic.py` 可设置输入文件夹、权重、输出目录与可选标签 TXT。结果为 predictions.csv，指定标签后计算 Macro F1 等指标。多模型投票使用 `infer/infer_vote.py` 和 `configs/inference/vote.yaml`；关闭的成员需训练并填写权重后再开启。

实验结果对比图运行 `python scripts/plot_f1_comparison.py`，路径在脚本底部修改。脚本核对各方法的验证划分，并从保存的投票预测 CSV 与测试标签重新计算测试 Macro F1；验证集与测试集分面展示，输出 PNG、SVG、PDF 和指标 CSV 到 `results/figures/f1_comparison_synthetic/`，无需重新训练或推理。

每次训练自动保存逐轮准确率/Macro F1 曲线、实际配置快照和日志；有验证集时额外保存每类指标、混淆矩阵、预测明细及特征 t-SNE。无验证集仅绘制训练曲线，分类结果保存为 train_ 数值文件。完整目录与 train.reporting 参数见 [训练结果输出](docs/TRAINING_OUTPUTS.md)。

## 有新数据集时怎么做

1. 将训练/验证总池整理为 `datasets/<数据集名>/<类别编号>/*.csv`，每个 CSV 是一条 CF/PW/PRI 序列。
2. 在数据集目录创建 data.yaml，指定 root、all、类别数和三个特征列。
3. 复制所需模型的 pipeline 和 train/infer 入口，修改数据路径、num_classes、结果目录、采样比例及序列长度。
4. 从头训练后，将推理 checkpoint 指向新权重；测试输入为平铺 CSV 文件夹，填写 labels.txt 路径后才评估。

完整目录、data.yaml 字段说明以及训练/推理范例见 [新数据集接入指南](docs/NEW_DATASET_GUIDE.md)。新小数据集建议先设 sample_ratio=1.0；显式代码参数优先于 YAML。已有数据无需运行生成器。

## 使用说明

- [数据集可视化：指定 data.yaml 查看原始序列与分布](docs/DATA_VISUALIZATION.md)
- [新数据集接入与 data.yaml 字段说明](docs/NEW_DATASET_GUIDE.md)
- [各模型入口](docs/MODEL_ENTRYPOINTS.md)
- [ResNet1D-18 + Softmax 模型与使用方法](docs/RESNET_SOFTMAX.md)
- [经典序列模型架构来源与配置说明](docs/CLASSIC_SEQUENCE_MODELS.md)
- [生成引擎](docs/SYNTHETIC_RADAR.md)
- [采样与训练/验证划分](docs/DATASET_SELECTION.md)
- [训练参数设置](docs/TRAIN_CONFIGURATION.md)
- [断点续训](docs/RESUME_TRAINING.md)
- [数据增强](docs/DATA_AUGMENTATION.md)
- [独立测试集、CSV 推理与投票](docs/INFERENCE_AND_TEST_DATA.md)

## 验证

```powershell
python -m pytest -q tests
```
