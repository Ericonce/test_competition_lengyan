# Models

本目录包含项目的信号分类模型，包括传统一维 CNN、EfficientNet、ResNet 以及序列模型。本 README 只介绍 `models/` 中可直接使用的实现；训练和推理入口见项目根目录 README。

## 1. 环境要求

```bash
# 仅适用于 CPU 或希望使用 PyPI 默认版本的环境
pip install torch
```

模型依赖 PyTorch。CUDA 和 NPU 环境请安装与当前 CUDA、CANN 及驱动匹配的 PyTorch 版本，并按根目录 README 配置 `torch_npu`（如需）。MambaSL 默认会自动选择后端：没有可用 CUDA 内核时使用纯 PyTorch 实现；也可以通过 `mamba_backend: pure` 或 `mamba_backend: ssm` 手动指定。

## 2. 输入和输出约定

所有模型都接收三维张量：

```text
inputs.shape == [batch_size, in_channels, sequence_length]
```

从 `MultiGrainCNN`、`ATENet`、`InvConvNet`、`MambaSL`、`ModernTCN` 和 `MPTSNet` 迁移的模型，`forward()` 返回一个字典，常用字段如下：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `logits` | `[B, num_classes]` | 分类 logits，训练时直接交给 `CrossEntropyLoss` |
| `stream_embedding` | `[B, D]` | 序列级特征 |
| `stream_contrast` | `[B, contrast_dim]` | 序列级对比学习特征 |
| `pulse_contrast` | `[B, T, contrast_dim]` | 时间步/脉冲级对比学习特征 |

只做分类时的基本用法：

```python
import torch
import torch.nn.functional as F
import models

model = models.build_model(
    {
        "type": "multigrain_cnn",
        "in_channels": 3,
    },
    num_classes=11,
)

x = torch.randn(8, 3, 256)
y = torch.randint(0, 11, (8,))
outputs = model(x)
loss = F.cross_entropy(outputs["logits"], y)
loss.backward()
```

推理时建议使用：

```python
model.eval()
with torch.no_grad():
    logits = model(x)["logits"]
    prediction = logits.argmax(dim=1)
```

## 3. 统一构建接口

`models.build_model(model_cfg, num_classes)` 根据配置中的 `type` 创建模型：

```python
from models import build_model

cfg = {
    "type": "mptsnet1d",
    "in_channels": 3,
    "d_model": 64,
    "depth": 2,
}
model = build_model(cfg, num_classes=11)
```

当前支持的 `type` 和别名如下：

| 模型 | `type` |
| --- | --- |
| CNN | `cnn` |
| EfficientNetB0 | `efficientnet`, `efficientnetb0`, `efficientnet-b0` |
| MultiGrainCNN | `cnn1d`, `multigrain_cnn` |
| ATENet | `atenet`, `atenet1d` |
| InvConvNet | `invconv1d`, `invconv` |
| MambaSL | `mambasl`, `mamba_sl`, `mamba` |
| ModernTCN | `modern_tcn`, `moderntcn` |
| MPTSNet | `mptsnet1d`, `mptsnet` |

`num_classes` 不放在配置字典中，而是作为 `build_model()` 的第二个参数传入。

### 3.1 通过 `utils.load_model()` 构建

如果使用项目现有的训练和推理脚本，可以直接调用 `utils.load_model()`。该函数接收模型名称、类别数、输入通道数、dropout、序列长度和目标设备：

```python
import utils

model = utils.load_model(
    "EfficientNetB0",
    class_num=11,
    in_channel=3,
    length=256,
    device="npu:0",
)
```

支持的名称包括：`ResNet18`、`CNN`、`EfficientNetB0`、`MultiGrainCNN`、`ATENet`、`InvConv`、`MambaSL`、`ModernTCN` 和 `MPTSNet`。名称不区分大小写，并支持 `cnn1d`、`mambasl`、`modern_tcn` 等别名。`device` 可以传入 `cpu`、`cuda:0`、`npu:0` 或 `torch.device`；训练和推理脚本会通过 `--device_type` 与 `--device` 自动解析设备。

## 4. 各模型使用方法

### 4.0 ResNet

`ResNet18` 是项目的一维残差网络，前向返回 `(feature, logits)`：

```python
import torch
from models import ResNet18

model = ResNet18(num_classes=11, in_channel=3)
x = torch.randn(8, 3, 256)
feature, logits = model(x)
```

### 4.1 CNN

`CNN` 是项目原有的多层一维卷积分类器，保留了固定大小的全连接分类头，直接返回 logits 张量，而不是输出字典。

```python
import torch
from models import CNN

model = CNN(num_classes=11, in_channel=3)
x = torch.randn(8, 3, 2728)
logits = model(x)                    # [8, 11]
features = model.feature(x)          # 展平后的卷积特征
```

该实现的 `fc1` 输入维度是固定的，因此输入序列长度必须与原始实现匹配；如果需要适配任意长度，建议使用 `MultiGrainCNN`。

### 4.2 EfficientNetB0

`EfficientNetB0` 是一维 EfficientNet，使用深度可分离卷积和 SE 模块，直接返回 logits 张量。

```python
import torch
from models import EfficientNetB0

model = EfficientNetB0(num_classes=11, in_channels=3)
x = torch.randn(8, 3, 256)
logits = model(x)                    # [8, 11]
features = model.feature(x)          # 全局池化后的特征
```

也可以通过统一接口构建：

```python
from models import build_model
model = build_model({"type": "efficientnetb0", "in_channels": 3}, num_classes=11)
```

`EfficientNetB0` 的输入长度可以变化，输入通道数必须与 `in_channels` 一致。

### 4.3 MultiGrainCNN

适合用多层一维卷积提取局部脉冲/序列模式。实现文件为 `models/multigrain_cnn.py`。

```python
from models import MultiGrainCNN

model = MultiGrainCNN(
    {
        "in_channels": 3,
        "hidden_channels": [64, 128, 256],
        "stream_embedding_dim": 256,
        "contrast_dim": 128,
        "dropout": 0.1,
        "pooling": "avg",       # "avg" 或 "avgmax"
    },
    num_classes=11,
)
```

主要配置：

- `in_channels`：输入通道数，默认 `2`。
- `hidden_channels`：卷积通道列表，默认 `[64, 128, 256]`。
- `stream_embedding_dim`：序列级特征维度。
- `contrast_dim`：对比学习投影维度，默认 `128`。
- `pooling`：`avg` 使用平均池化；`avgmax` 拼接平均池化和最大池化结果。
- `dropout`：分类/投影头 dropout 比例。

### 4.4 ATENet

ATENet 将一个通道作为相对观测时间，并通过时间注意力聚合序列。如果输入已经包含时间通道，默认使用索引 `2`；如果输入通道数不超过 `time_channel`，模型会自动补充 `[0, 1]` 归一化时间通道。

```python
from models import ATENetClassifier

model = ATENetClassifier(
    {
        "in_channels": 3,
        "hidden": 64,
        "embed_time": 32,
        "num_heads": 4,
        "time_channel": 2,
        "contrast_dim": 128,
        "dropout": 0.1,
    },
    num_classes=11,
)
```

`embed_time` 必须能被 `num_heads` 整除。若使用三通道 PDW 输入，通常将时间特征放在第 3 个通道，即 `time_channel=2`。

### 4.5 InvConvNet

InvConvNet 使用标准卷积、局部偏移不变卷积和局部线性趋势不变卷积三个分支，适合对局部统计变化较敏感的序列。

```python
from models import InvConvClassifier

model = InvConvClassifier(
    {
        "in_channels": 3,
        "d_model": 64,
        "kernel_size": 51,
        "sequence_length": 256,
        "contrast_dim": 128,
        "dropout": 0.2,
    },
    num_classes=11,
)
```

`kernel_size` 必须为奇数；模型会根据 `sequence_length` 自动限制过大的卷积核。`d_model` 至少为 `3`。

### 4.6 MambaSL

MambaSL 先将 `[B, C, L]` 转换为 `[B, L, C]`，然后进行状态空间序列建模。

```python
from models import MambaSLClassifier

model = MambaSLClassifier(
    {
        "in_channels": 3,
        "d_model": 128,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "stream_embedding_dim": 128,
        "contrast_dim": 128,
        "dropout": 0.1,
        "mamba_backend": "auto",
    },
    num_classes=11,
)
```

`mamba_backend` 可选：

- `auto`：自动选择；
- `pure`：使用纯 PyTorch 实现，兼容性最好；
- `ssm`：优先使用 `mamba-ssm`，不可用时自动回退到 `pure`。

### 4.7 ModernTCN

ModernTCN 使用大卷积核、小卷积核和通道/变量混合模块。输入长度不需要固定为配置示例中的长度，但至少应能容纳一个 patch。

```python
from models import ModernTCNClassifier

model = ModernTCNClassifier(
    {
        "in_channels": 3,
        "d_model": 32,
        "depth": 2,
        "large_kernel": 31,
        "small_kernel": 5,
        "patch_size": 8,
        "patch_stride": 4,
        "ffn_ratio": 2,
        "contrast_dim": 128,
        "dropout": 0.1,
    },
    num_classes=11,
)
```

`in_channels` 必须与输入张量的第 2 维一致。`large_kernel` 和 `small_kernel` 通常使用奇数，以保持序列长度稳定。

### 4.8 MPTSNet

MPTSNet 融合多尺度局部卷积、周期混合、全局 token 混合和统计池化，支持长度变化的输入序列。

```python
from models import MPTSNetClassifier

model = MPTSNetClassifier(
    {
        "in_channels": 3,
        "d_model": 64,
        "depth": 2,
        "kernel_sizes": [3, 7, 15, 31],
        "periods": [2, 4, 8, 16],
        "num_heads": 4,
        "token_count": 32,
        "norm": "batch",
        "contrast_dim": 128,
        "dropout": 0.1,
    },
    num_classes=11,
)
```

注意事项：

- `d_model` 必须能被 `num_heads` 整除；
- `kernel_sizes` 中的值必须为奇数；
- `periods` 中的值必须大于 `1`；
- 输入序列长度至少为 `2`；
- `norm` 支持 `batch` 和 `group`。

## 5. 从 YAML 配置构建

项目没有在 `models` 包内提供 YAML 加载器，但可以使用 `yaml.safe_load` 后传给统一构建函数：

```python
from pathlib import Path
import yaml
from models import build_model

with Path("model.yaml").open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 配置文件中的 name 字段不会参与模型构造，type 和其余字段会传入模型。
model = build_model(cfg, num_classes=11)
```

示例 `model.yaml`：

```yaml
type: modern_tcn
in_channels: 3
d_model: 32
depth: 2
large_kernel: 31
small_kernel: 5
patch_size: 8
patch_stride: 4
ffn_ratio: 2
dropout: 0.1
contrast_dim: 128
```

## 6. 训练和保存

训练时通常只对 `outputs["logits"]` 计算分类损失：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

model.train()
outputs = model(inputs)
loss = torch.nn.CrossEntropyLoss()(outputs["logits"], labels.long())
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

保存和加载：

```python
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
```

如果训练中还使用对比学习损失，可以额外使用 `stream_contrast` 和 `pulse_contrast`；纯分类任务不需要使用这些字段。

## 7. 与 ResNet 的区别

本次迁移的模型统一返回输出字典。现有 `models/resnet.py` 中的 ResNet 保持原有接口，前向结果仍为：

```python
feature, logits = resnet_model(inputs)
```

因此，使用新模型时应读取 `outputs["logits"]`，不要按 ResNet 的元组接口解包。

## 8. Python 文件索引

顶层模型文件：

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 导出模型并提供 `MODEL_REGISTRY`、`build_model()`。 |
| `resnet.py` | 一维 ResNet 基础块、瓶颈块及不同深度构造函数。 |
| `cnn.py` | 固定卷积层与全连接头的一维 CNN。 |
| `efficientnet.py` | 一维 EfficientNet、MBConv 和 SE 模块。 |
| `multigrain_cnn.py` | MultiGrainCNN 的兼容导出入口。 |
| `atenet.py` | ATENet 分类器的兼容导出入口。 |
| `invconv.py` | InvConv 分类器的兼容导出入口。 |
| `mambasl.py` | MambaSL 分类器的兼容导出入口。 |
| `modern_tcn.py` | ModernTCN 分类器的兼容导出入口。 |
| `mptsnet.py` | MPTSNet 分类器的兼容导出入口。 |

`nn/` 内部实现：

| 文件 | 作用 |
| --- | --- |
| `nn/__init__.py` | 汇总导出迁移后的分类器。 |
| `nn/multigrain_cnn.py` | MultiGrainCNN 的完整网络实现。 |
| `nn/classifiers/__init__.py` | 汇总导出分类器封装。 |
| `nn/classifiers/outputs.py` | 统一组织 logits、序列特征和对比特征输出。 |
| `nn/classifiers/atenet.py` | ATENet 模块的分类任务封装。 |
| `nn/classifiers/invconv.py` | InvConv 模块的分类任务封装。 |
| `nn/classifiers/mambasl.py` | Mamba 模块的分类任务封装。 |
| `nn/classifiers/modern_tcn.py` | ModernTCN 模块的分类任务封装。 |
| `nn/classifiers/mptsnet.py` | MPTSNet 模块的分类任务封装。 |
| `nn/modules/__init__.py` | 汇总导出底层特征模块。 |
| `nn/modules/atenet.py` | 时间嵌入与注意力编码模块。 |
| `nn/modules/invconv.py` | 局部偏移/趋势不变卷积模块。 |
| `nn/modules/mamba.py` | Mamba 序列模块及纯 PyTorch 回退实现。 |
| `nn/modules/modern_tcn.py` | 大核时序卷积和 patch 模块。 |
| `nn/modules/mptsnet.py` | 多尺度、周期与 token 混合模块。 |
| `nn/heads/__init__.py` | 汇总导出任务头。 |
| `nn/heads/projection.py` | 序列级和脉冲级投影头。 |
| `nn/heads/adversarial.py` | 梯度反转与对抗分类头。 |

通常只需从 `models` 或 `utils.load_model()` 构建完整分类器，不需要直接实例化 `nn/modules` 与 `nn/heads` 中的内部组件。
