# 合成数据集的模型训练与推理入口

所有方法输入均为 CF/PW/PRI，默认使用 datasets/synthetic_radar_overlap/data.yaml 中记录的合成总池。其他三通道数据可按 [新数据集接入指南](NEW_DATASET_GUIDE.md) 创建独立 pipeline 与入口，表中为现有 synthetic 默认配置。

| 方法 | 流程配置 | 训练 | 推理 |
| --- | --- | --- | --- |
| cnn_softmax | [yaml](../configs/pipeline/cnn_softmax_synthetic.yaml) | [train.py](../train/cnn_softmax_synthetic/train.py) | [infer.py](../infer/cnn_softmax_synthetic/infer.py) |
| resnet_softmax | [yaml](../configs/pipeline/resnet_softmax_synthetic.yaml) | [train.py](../train/resnet_softmax_synthetic/train.py) | [infer.py](../infer/resnet_softmax_synthetic/infer.py) |
| resnet34 | [yaml](../configs/pipeline/resnet34_synthetic.yaml) | [train.py](../train/resnet34_synthetic/train.py) | [infer.py](../infer/resnet34_synthetic/infer.py) |
| resnet50 | [yaml](../configs/pipeline/resnet50_synthetic.yaml) | [train.py](../train/resnet50_synthetic/train.py) | [infer.py](../infer/resnet50_synthetic/infer.py) |
| tcn | [yaml](../configs/pipeline/tcn_synthetic.yaml) | [train.py](../train/tcn_synthetic/train.py) | [infer.py](../infer/tcn_synthetic/infer.py) |
| inception_time | [yaml](../configs/pipeline/inception_time_synthetic.yaml) | [train.py](../train/inception_time_synthetic/train.py) | [infer.py](../infer/inception_time_synthetic/infer.py) |
| lstm | [yaml](../configs/pipeline/lstm_synthetic.yaml) | [train.py](../train/lstm_synthetic/train.py) | [infer.py](../infer/lstm_synthetic/infer.py) |
| bilstm | [yaml](../configs/pipeline/bilstm_synthetic.yaml) | [train.py](../train/bilstm_synthetic/train.py) | [infer.py](../infer/bilstm_synthetic/infer.py) |
| transformer | [yaml](../configs/pipeline/transformer_synthetic.yaml) | [train.py](../train/transformer_synthetic/train.py) | [infer.py](../infer/transformer_synthetic/infer.py) |
| informer | [yaml](../configs/pipeline/informer_synthetic.yaml) | [train.py](../train/informer_synthetic/train.py) | [infer.py](../infer/informer_synthetic/infer.py) |
| mlp | [yaml](../configs/pipeline/mlp_synthetic.yaml) | [train.py](../train/mlp_synthetic/train.py) | [infer.py](../infer/mlp_synthetic/infer.py) |
| atenet | [yaml](../configs/pipeline/atenet_synthetic.yaml) | [train.py](../train/atenet_synthetic/train.py) | [infer.py](../infer/atenet_synthetic/infer.py) |
| invconv1d | [yaml](../configs/pipeline/invconv1d_synthetic.yaml) | [train.py](../train/invconv1d_synthetic/train.py) | [infer.py](../infer/invconv1d_synthetic/infer.py) |
| mambasl | [yaml](../configs/pipeline/mambasl_synthetic.yaml) | [train.py](../train/mambasl_synthetic/train.py) | [infer.py](../infer/mambasl_synthetic/infer.py) |
| modern_tcn | [yaml](../configs/pipeline/modern_tcn_synthetic.yaml) | [train.py](../train/modern_tcn_synthetic/train.py) | [infer.py](../infer/modern_tcn_synthetic/infer.py) |
| mptsnet1d | [yaml](../configs/pipeline/mptsnet1d_synthetic.yaml) | [train.py](../train/mptsnet1d_synthetic/train.py) | [infer.py](../infer/mptsnet1d_synthetic/infer.py) |
| multigrain_infonce | [yaml](../configs/pipeline/multigrain_infonce_synthetic.yaml) | [train.py](../train/multigrain_infonce_synthetic/train.py) | [infer.py](../infer/multigrain_infonce_synthetic/infer.py) |
| pulse_hypothesis_fused | [yaml](../configs/pipeline/pulse_hypothesis_fused_synthetic.yaml) | [train.py](../train/pulse_hypothesis_fused_synthetic/train.py) | [infer.py](../infer/pulse_hypothesis_fused_synthetic/infer.py) |
| modular_multigrain_iq | [yaml](../configs/pipeline/modular_multigrain_iq_synthetic.yaml) | [train.py](../train/modular_multigrain_iq_synthetic/train.py) | [infer.py](../infer/modular_multigrain_iq_synthetic/infer.py) |

另有 `cnn_softmax_synthetic_baseline`，作为同一总池上的独立 CNN 调参配置。

resnet_softmax 的全名为 ResNet1D-18 + Softmax，使用残差编码器、全局平均池化和线性分类层，训练采用交叉熵。结构与参数说明见 [ResNet 分类模型](RESNET_SOFTMAX.md)。

各 train/infer 入口直接列出路径、类别数及常用参数；pipeline 提供其他默认设置，调用中的参数优先。采样、划分、输入长度、补长策略和进度可直接在调用中调整。新增 synthetic 配置保留各方法的网络和损失定义。新增 ResNet1D-34/50、TCN、InceptionTime、LSTM/BiLSTM、Transformer Encoder、Informer Encoder 与 MLP，架构来源及分类适配说明见[经典序列模型说明](CLASSIC_SEQUENCE_MODELS.md)。

训练输出在 model.train() 的 output_dir 设置；各推理入口显式填写 checkpoint 和 save_dir，默认使用对应训练目录的 best_model.pt 与 infer_test 子目录。

各训练入口自动生成逐轮指标曲线、配置快照和日志，并在有验证集时生成每类指标图、混淆矩阵及 t-SNE。目录与设置见 [训练结果输出](TRAINING_OUTPUTS.md)。

路径和常用参数均写在 model.infer() 内的范例为 [infer_cnn_softmax_synthetic.py](../infer/infer_cnn_softmax_synthetic.py)。多模型投票为 [infer_vote.py](../infer/infer_vote.py)，成员配置只引用 synthetic pipeline。
