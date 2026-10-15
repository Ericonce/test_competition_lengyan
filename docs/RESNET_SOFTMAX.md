# 一维残差网络分类模型（ResNet1D-18 + Softmax）

项目已有 ResNet1D 骨干和使用它的 Modular 模型。新增的独立方法名为 **resnet_softmax**，完整名称是 **ResNet1D-18 + Softmax 脉冲序列分类模型**。

## 网络与训练方式

输入为三通道 CF/PW/PRI，模型沿脉冲序列使用一维卷积。复用项目自己的 ResNet1D-18 骨干，四个残差阶段各有两个基本块：

```text
[B, 3, L]
  → 一维 ResNet-18 骨干
[B, 512, 下采样长度]             # base_channels=64 时
  → 全局平均池化
[B, 512]
  → Dropout
  → 全连接分类层
[B, num_classes] logits
  → 推理时 Softmax
[B, num_classes] 类别概率
```

该方法使用交叉熵进行监督分类，模型输出原始 logits，交叉熵内部处理 log-softmax；推理的公共 predictor 再使用 Softmax 计算概率和 confidence，避免训练时重复执行 Softmax。

它是独立的 ResNet 分类基线，网络只包含残差编码器、池化、Dropout 和分类层。新 pipeline 的流级、脉冲级对比和对抗损失权重均为 0。数据增强仍可按训练入口和增强 YAML 开启。

## 文件位置

| 用途 | 文件 |
| --- | --- |
| 模型实现 | [resnet_softmax.py](../pulsecls/nn/classifiers/resnet_softmax.py) |
| 网络结构设置 | [resnet_softmax.yaml](../configs/model/resnet_softmax.yaml) |
| 训练/推理 pipeline | [resnet_softmax_synthetic.yaml](../configs/pipeline/resnet_softmax_synthetic.yaml) |
| 独立训练入口 | [train.py](../train/resnet_softmax_synthetic/train.py) |
| 独立推理入口 | [infer.py](../infer/resnet_softmax_synthetic/infer.py) |

模型注册名为 resnet_softmax，也支持 resnet1d_softmax 别名。现有 resnet1d 和 Modular 注册名继续对应原实现。

## 训练与推理

从项目根目录运行：

```bash
python train/resnet_softmax_synthetic/train.py
python infer/resnet_softmax_synthetic/infer.py
```

默认训练数据是 datasets/synthetic_radar_overlap/data.yaml，类别数 100，输入长度 256，抽取总池的 1% 后按 80%/20% 分训练/验证。默认训练 40 轮、batch_size=64、lr=0.001、patience=6，输出 results/resnet_softmax_synthetic。

训练入口直接列出上述路径、类别数、采样/划分、增强 YAML、设备和续训等常用设置。显式调用参数优先于 YAML；网络宽度 base_channels 和 dropout 在模型 YAML 修改。基础宽度 64 对应阶段通道 64、128、256、512，宽度调小可减少参数和计算量。

推理入口显式选择 best_model.pt 和测试 CSV 文件夹，输出 infer_test/predictions.csv。labels_file="" 时只预测，填写 labels.txt 路径后计算 Macro-F1 等指标。无验证的全量训练使用 final_model.pt；需要续训时指定 last_model.pt，resume=True 后 epochs 表示额外轮数。

换成小数据集或新数据集时同步修改训练 data、num_classes、output_dir，以及推理 source、checkpoint、save_dir 和 num_classes。小数据集可将 sample_ratio 改为 1.0，避免样本太少。完整说明见 [新数据集接入](NEW_DATASET_GUIDE.md) 和 [训练参数](TRAIN_CONFIGURATION.md)。

设备 auto 沿用项目的 CUDA → 昇腾 NPU → CPU 顺序。模型使用 PyTorch 的 Conv1d、BatchNorm1d、ReLU、池化和线性层，无需安装新的模型库或下载预训练权重。

## 加入多模型投票

infer/infer_vote.py 和 configs/inference/vote.yaml 已添加 resnet_softmax 成员。先在与其他成员相同的辐射源及标签编号上训练，再在 model.infer_vote() 调用中设置 enabled=True 并填写对应 checkpoint；调用中的成员列表优先于投票 YAML。不参与时设置 enabled=False。规则见 [投票推理](INFERENCE_AND_TEST_DATA.md)。
