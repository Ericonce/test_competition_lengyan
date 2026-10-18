# 雷达型号识别

本项目使用雷达脉冲描述字（PDW）序列完成型号分类，支持原始 CSV 划分、定长切片、按通道归一化、数据增强、多种一维分类模型，以及 CPU、CUDA 和华为昇腾 NPU 训练与推理。

## Python 文件

| 文件 | 作用 |
| --- | --- |
| `pretrain.py` | 使用两路 PDW 数据增强和对比损失，对 `models/` 中的分类模型进行自监督预训练。 |
| `train.py` | 构建训练/验证/测试数据集，训练模型，统计切片级或源文件级指标，保存权重、分类准确率和混淆矩阵。 |
| `inference.py` | 加载测试目录和训练权重，执行测试，按自然顺序将预测标签保存为 CSV。 |
| `Inference_filter_rf.py` | 从已知型号数据学习典型 RF 分布，先排除 RF 不匹配型号，再修正模型推理结果。 |
| `utils.py` | 数据兼容接口、设备解析和模型统一加载入口。 |

数据处理脚本（划分、切片、CSV 特征解析、数据集读取、统计和可视化）的参数与数据格式说明见 `data_process/README.md`。

数据可视化：`data_process/visualize_pdw.py` 会启动 Plotly 网页，递归列出指定目录下的 PDW CSV。选择文件后可查看载频、脉宽、PRI 的时序散点图，并通过 X/Y 下拉框查看脉宽-载频、脉宽-PRI、载频-PRI 等二维散点图：

```bash
python data_process/visualize_pdw.py --input_dir ./dataset/cut_data
```

子目录说明见：

- `data_process/README.md`：数据划分、切片、读取、归一化与增强。
- `models/README.md`：全部模型、构建接口和模型文件索引。
- `tool/README.md`：日志、配置保存和通用辅助类。

## 环境

基础依赖包括 Python、PyTorch、NumPy、pandas、h5py、scikit-learn、SciPy、Matplotlib 和 tqdm。CUDA 环境安装匹配 CUDA 的 PyTorch；昇腾 910B3 环境还需安装与 CANN、PyTorch 版本匹配的 `torch_npu`。

```bash
pip install numpy pandas h5py scikit-learn scipy matplotlib tqdm
```

PyTorch 请根据 CPU、CUDA 或 NPU 环境从对应官方渠道安装，不能用一条通用命令覆盖所有硬件。

## 完整训练流程

1. 使用 `data_process/split_data.py` 在切片前划分训练集、验证集和测试集。路径清单默认保存到 `dataset/split_data/`：

   ```bash
   python data_process/split_data.py --data_dir ./raw_data --output_dir ./dataset/split_data
   ```

2. 推荐使用 `data_process/cut_data_multi.py` 对三个清单分别切片。它默认输出 H5，但默认文件名是 `val.h5`，因此示例显式指定 `--h5_name data.h5`。单进程 `cut_data.py` 默认输出旧 CSV 格式；若要输出 H5，需显式传入 `--output_format h5 --h5_name data.h5`。数据量较大时建议使用多进程版本：

   ```bash
   python data_process/cut_data_multi.py --file_list ./dataset/split_data/train.txt --output_dir ./dataset/cut_data/train --slice_len 256 --workers 8 --h5_name data.h5
   python data_process/cut_data_multi.py --file_list ./dataset/split_data/val.txt --output_dir ./dataset/cut_data/val --slice_len 256 --workers 8 --h5_name data.h5
   python data_process/cut_data_multi.py --file_list ./dataset/split_data/test.txt --output_dir ./dataset/cut_data/test --slice_len 256 --workers 8 --h5_name data.h5
   ```

3. 可选：使用 `pretrain.py` 进行自监督预训练：

   ```bash
   python pretrain.py --data_root ./dataset/cut_data --model ResNet18 --sequence_length 256 --epochs 200
   ```

4. 使用 `train.py` 进行监督训练。可以直接训练，也可以通过 `--pretrained` 和 `--pretrained_path` 加载预训练权重：

   ```bash
   python train.py --data_root ./dataset/cut_data --model ResNet18 --sequence_length 256

   python train.py --data_root ./dataset/cut_data --model ResNet18 --sequence_length 256 --pretrained --pretrained_path ./result/pretrain/ResNet18/radar_0/weights/pretrain_best.pth
   ```

5. 使用 `inference.py` 进行推理测试：

   ```bash
   python inference.py --data_root ./dataset/cut_data/test --model ResNet18 --session_name radar_0
   ```

   默认预测结果保存到：

   ```text
   result/cut_data/{model_name}/{session_name}/predictions.csv
   ```

## 数据流程

原始数据按类别目录组织：

```text
raw_data/
├── 0/*.csv
├── 1/*.csv
└── ...
```

先按原始文件划分，避免同一信号的不同切片泄漏到不同数据集：

```bash
python data_process/split_data.py --data_dir ./raw_data --output_dir ./dataset/split_data
```

再分别切片，推荐使用多进程版本：

```bash
python data_process/cut_data_multi.py --file_list ./dataset/split_data/train.txt --output_dir ./dataset/cut_data/train --slice_len 256 --workers 8 --h5_name data.h5
python data_process/cut_data_multi.py --file_list ./dataset/split_data/val.txt --output_dir ./dataset/cut_data/val --slice_len 256 --workers 8 --h5_name data.h5
python data_process/cut_data_multi.py --file_list ./dataset/split_data/test.txt --output_dir ./dataset/cut_data/test --slice_len 256 --workers 8 --h5_name data.h5
```

推荐的训练数据结构如下：

```text
dataset/cut_data/
├── train/data.h5
├── val/data.h5
└── test/data.h5
```

H5 内保存 `float32` 切片、类别标签和原始文件分组，特征列默认为 `CF_MHz`、`PW_us`、`PRI_us`。相较于大量小 CSV，H5 避免了逐文件打开和文本解析，尤其适合数十万切片。`dataset.py` 仍兼容旧的 `data_root/{train,val,test}/{类别名}/*.csv` 结构；同一目录同时存在 H5 和 CSV 时优先读取 H5。

训练或推理时可用 `--data_format` 显式选择格式：`auto`（优先 H5）、`h5`（强制 H5）或 `csv`（强制逐文件读取 CSV）。`pretrain.py` 和 `inference.py` 默认使用 `auto`，`train.py` 为兼容旧 CSV 流程默认使用 `csv`；建议显式指定。例如：

```bash
python train.py --data_root ./dataset/cut_data --data_format h5
python train.py --data_root ./dataset/cut_data --data_format csv
python inference.py --data_root ./dataset/cut_data/test --data_format h5
```

CSV 特征列由 `data_process/csv_features.py` 统一解析。带表头时会按
`CF_MHz/RF`、`PW_us/PW`、`PRI_us/PRI` 及常见中英文别名识别，列顺序可以变化。
没有可识别表头时，纯数字本身无法可靠说明哪一列是 RF、PW 或 PRI，需显式指定物理顺序：

```bash
python train.py --data_format csv --no-header-order PRI,RF,PW
python inference.py --data_format csv --no-header-order PRI,RF,PW
```

不指定无表头顺序时，程序会报错而不会静默交换特征。

如需继续生成旧格式，显式指定：

```bash
python data_process/cut_data_multi.py --file_list ./dataset/split_data/train.txt --output_dir ./dataset/cut_data/train --slice_len 256 --workers 8 --output_format csv
```

当前数据集加载器会在训练开始时将一个 split 的 H5 数据整批读入内存，以获得较高的训练迭代速度，因此内存容量仍需覆盖切片数组和归一化后的数据。

## 自监督预训练

预训练使用与监督训练相同的 `data_root` 和按通道归一化方式，但不使用类别标签。默认使用 hard-negative 对比损失：

```bash
python pretrain.py --data_root ./dataset/cut_data --model ResNet18 --sequence_length 256 --epochs 200 --device_type auto --device 0
```

可通过 `--loss {hard_negative,nt_xent}` 选择损失，并通过 `--augment_methods` 配置两路随机增强。默认输出目录为：

```text
result/pretrain/<模型>/<session_name>_<序号>/
├── parameters.txt
├── epoch_times.txt
├── <session_name>_<序号>.log
└── weights/
    ├── pretrain_best.pth
    └── pretrain_last.pth
```

预训练检查点中的 `state_dict` 包含实际参与自监督训练的编码器参数和模型缓冲区，不包含未经训练的分类头；`projector_state_dict` 是只用于对比学习的外部投影头。

## 训练

```bash
python train.py --data_root ./dataset/cut_data --model ResNet18 --sequence_length 256 --device_type auto --device 0
```

加载预训练权重进行微调：

```bash
python train.py --data_root ./dataset/cut_data --model ResNet18 --sequence_length 256 --pretrained --pretrained_path ./result/pretrain/ResNet18/radar_0/weights/pretrain_best.pth --device_type auto --device 0
```

预训练模型与监督模型应使用相同的 `--model`。加载器只载入名称和形状都兼容的参数，不匹配的分类头会自动跳过。

常用选项：

- `--fusion {none,vote,score,probability}`：验证和测试的聚合方式；默认 `probability`。
- `--augment` / `--no-augment`：开关数据增强。
- `--mixup` / `--no-mixup`：开关 Manifold MixUp；启用时在最终分类头之前的特征表示上混合，不改变原始输入。
- `--label_smoothing` / `--no-label-smoothing`：开关标签平滑。
- `--normalization {none,standard,minmax}`：按特征通道归一化。
- `--device_type {auto,npu,cuda,cpu}` 和 `--device`：选择计算设备。
- `--pretrained` / `--no-pretrained`：开关自监督预训练权重加载。
- `--pretrained_path`：`pretrain_best.pth` 或 `pretrain_last.pth` 路径。

融合方式含义：

| 值 | 统计粒度 |
| --- | --- |
| `none` | 每个切片独立预测。 |
| `vote` | 同一源文件的切片类别做硬投票。 |
| `score` | 同一源文件的原始 logits 求均值后分类。 |
| `probability` | logits 经 softmax 后按源文件求概率均值再分类。 |

训练集始终统计切片级指标；`--fusion` 只控制验证集和测试集。输出指标包含总体准确率、平均类别准确率、weighted-F1 和 macro-F1。每个 epoch 的耗时同时写入训练日志、检查点的 `epoch_seconds` 字段和实验目录下的 `epoch_times.txt`。

## 推理

```bash
python inference.py --data_root ./dataset/cut_data/test --model ResNet18 --session_name radar_0 --fusion probability --device_type auto --device 0
```

默认读取 `result/cut_data/<模型>/<session_name>/weights/model_best.pth`，并在实验目录生成 `predictions.csv`。CSV 第一列为自然排序后的序号，第二列为预测标签。使用 `--fusion none` 时每个切片输出一行；其他模式每个原始源文件输出一行。

### RF 粗分类修正推理

使用原始、未切片、未划分的 CSV 数据建立各型号的多峰 RF 典型分布，然后在 RF 兼容候选型号中选择模型概率最高者。参考目录必须按类别组织，例如 `raw_data/0/*.csv`、`raw_data/1/*.csv`。参考 CSV 仍由共享解析器读取，因此必须包含可识别的 RF/CF、PW 和 PRI 三列（程序只使用其中的 RF/CF 数值）；有表头时列顺序可变，无表头时必须传入 `--no-header-order`。测试数据仍读取切片后的 CSV 或 H5：

```bash
python Inference_filter_rf.py --reference_root ./raw_data --test_root ./dataset/cut_data/test --model ResNet18 --session_name radar_0 --test_format h5 --rf_workers 8
```

默认 `--rf_mode hard` 会排除 RF 匹配比例低于 `--rf_match_threshold 0.5` 的型号；`--rf_tolerance_mhz` 控制典型频段两侧容差。输出除 `predictions_rf_filtered.csv` 外，还包括 `rf_profiles.csv`、`rf_profile_heatmap.png` 和逐条修正原因表 `rf_filter_details.csv`。若某个型号在参考数据中不存在 RF 统计，该型号默认不会被过滤。

首次运行会统计原始参考 CSV 并默认在 `reference_root` 旁生成 `_rf_profile_cache.npz`。后续运行会在参考目录签名、类别列表和 RF 参数一致时直接加载缓存，不再读取原始 CSV。可以指定缓存位置或强制重建：

```bash
python Inference_filter_rf.py ... --rf_profile_cache ./dataset/rf_profiles.npz
python Inference_filter_rf.py ... --rebuild_rf_profile
```

完整参数可运行：

```bash
python train.py --help
python pretrain.py --help
python inference.py --help
```
