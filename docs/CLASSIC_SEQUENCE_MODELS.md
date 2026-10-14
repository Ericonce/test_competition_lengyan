# 新增经典序列分类模型

本项目新增九个三通道序列分类模型，输入为 `(batch, 3, sequence_length)`，统一返回 `logits` 与 `stream_embedding`。没有引入额外第三方模型依赖；模型均由 PyTorch 层组成，类别数由现有 pipeline/train 配置决定。

| 模型 | 项目实现 | 关键配置 | 用途与结构 |
| --- | --- | --- | --- |
| ResNet1D-34 | `configs/model/resnet34.yaml` | `base_channels`, `dropout` | BasicBlock 残差阶段 `[3,4,6,3]`，一维卷积版本。 |
| ResNet1D-50 | `configs/model/resnet50.yaml` | `base_channels`, `dropout` | Bottleneck 残差阶段 `[3,4,6,3]`，一维卷积版本；stride 放在 3-tap 卷积中。 |
| TCN | `configs/model/tcn.yaml` | `channels`, `kernel_size`, `dropout` | 左侧填充保持因果性，膨胀率按层翻倍，残差聚合后做时间平均池化。 |
| InceptionTime | `configs/model/inception_time.yaml` | `depth`, `filters`, `bottleneck`, `dropout` | 并行多尺度卷积加池化分支，每三个模块做一次残差连接。 |
| LSTM | `configs/model/lstm.yaml` | `hidden_size`, `num_layers`, `dropout` | 单向 LSTM，以最后时刻的隐藏状态分类。 |
| BiLSTM | `configs/model/bilstm.yaml` | `hidden_size`, `num_layers`, `dropout` | 双向 LSTM，拼接前向和后向最终隐藏状态。 |
| Transformer | `configs/model/transformer.yaml` | `d_model`, `n_heads`, `num_layers`, `dim_feedforward` | 特征投影、可学习位置编码、Transformer Encoder 和时间平均池化。 |
| Informer | `configs/model/informer.yaml` | `d_model`, `n_heads`, `e_layers`, `d_ff`, `factor`, `distil` | ProbSparse 注意力 Encoder，可选卷积蒸馏；末端平均池化用于分类。 |
| MLP | `configs/model/mlp.yaml` | `hidden_dims`, `dropout` | 将固定长度三通道序列展平后输入全连接网络。 |

ResNet 的残差阶段与 Bottleneck 设计参考 [torchvision ResNet 实现](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)。此处将图像二维卷积改为适用于 `(CF, PW, PRI)` 的一维卷积；网络深度和残差结构保留，通道宽度由 `base_channels` 控制。TCN 参考 [Locuslab TCN 官方实现](https://github.com/locuslab/TCN/blob/master/TCN/tcn.py) 的两层膨胀因果卷积残差块，使用左侧补零实现与 Chomp 等价的因果裁剪。InceptionTime 参考[作者仓库中的 InceptionTime 分类网络](https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py)的瓶颈投影、多尺度卷积分支和残差连接；本项目是独立的 PyTorch 适配实现，并未复制整份外部代码。

LSTM/BiLSTM 使用 [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTM.html)。Transformer 使用 [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.transformer.TransformerEncoder.html)。Informer 的 ProbSparse 注意力和蒸馏思路参考[作者仓库的注意力模块](https://github.com/zhouhaoyi/Informer2020/blob/main/models/attn.py)及[Encoder 实现](https://github.com/zhouhaoyi/Informer2020/blob/main/models/encoder.py)。原版 Informer 面向长序列预测并含解码器；本项目使用其 ProbSparse Encoder 设计，接入分类池化层和分类器，不包含预测解码器。

每个模型都有独立的 synthetic pipeline、训练入口和推理入口：

```powershell
python .\train\resnet34_synthetic\train.py
python .\train\informer_synthetic\train.py
python .\infer\resnet34_synthetic\infer.py
python .\infer\informer_synthetic\infer.py
```

常调训练参数在 `train.py` 的 `model.train()` 调用中；网络超参数在对应 `configs/model/*.yaml` 中。入口默认读取 synthetic 合成数据集。推理前需要先训练该模型，并确认入口内的 checkpoint 指向实际生成的权重。模型以统一分类输出接口接入，不代表已在所有 CUDA/NPU 型号和所有数据集上完成运行验证。
