# 数据处理

本目录负责原始 CSV 的划分、切片、统计，以及训练时的数据读取、归一化和增强。标签统一取自类别文件夹名称。

## Python 文件

| 文件 | 作用 |
| --- | --- |
| `split_data.py` | 按类别分别打乱原始 CSV，并生成 `train.txt`、`val.txt`、`test.txt`，不复制原文件。 |
| `cut_data.py` | 单进程滑动窗口切片；短序列循环填充，默认输出 CSV，传入 `--output_format h5` 才写入 H5。 |
| `cut_data_multi.py` | 多进程读取和切片、主进程安全写入 H5；失败文件写入 `_errors.txt`。 |
| `signal_length_stat.py` | 递归统计原始 CSV 的序列长度并生成分布图。 |
| `h5_storage.py` | 定义切片 H5 的路径发现、数据结构、缓冲追加与压缩写入。 |
| `csv_features.py` | 统一识别有表头或无表头 CSV 的 RF/CF、PW、PRI 列，并规范化为固定模型输入顺序。 |
| `dataset.py` | 优先读取 H5、兼容旧 CSV，保留源文件分组，并使用训练集统计量做按通道归一化。 |
| `augment.py` | 雷达 PDW 序列增强，包括循环移位、高斯噪声、缩放、通道遮挡、脉冲遮挡和反转。 |
| `visualize_pdw.py` | 启动 Plotly 网页，递归浏览目录下的 CSV，并绘制三组时序散点图和可选 X/Y 轴的二维散点图。 |
| `analyze_pdw_features.py` | 按型号统计载频、脉宽和 PRI，并生成连续样本图、分布图、箱线图和型号间分布差异热力图。 |

## 1. 划分原始数据

输入结构：

```text
raw_data/
├── 0/*.csv
├── 1/*.csv
└── ...
```

运行：

```bash
python data_process/split_data.py --data_dir ./raw_data --output_dir ./dataset/split_data --train 0.8 --val 0.1 --test 0.1 --seed 42
```

每行清单格式为：

```text
绝对或相对CSV路径,类别名
```

## 2. 切片

推荐多进程版本：

```bash
python data_process/cut_data_multi.py --file_list ./dataset/split_data/train.txt --output_dir ./dataset/cut_data/train --slice_len 256 --overlap 0 --workers 8 --chunksize 1 --h5_name data.h5
```

单进程版本：

```bash
python data_process/cut_data.py --file_list ./dataset/split_data/train.txt --output_dir ./dataset/cut_data/train --slice_len 256 --overlap 0 --output_format h5 --h5_name data.h5
```

也可以不提供清单，直接让多进程脚本递归扫描 `input_dir` 下的类别目录：

```bash
python data_process/cut_data_multi.py --input_dir ./raw_data --file_list "" --output_dir ./dataset/cut_data/train --slice_len 256 --h5_name data.h5
```

多进程脚本默认 `--output_format h5`，但默认 H5 文件名为 `val.h5`；上面的示例用 `--h5_name data.h5` 统一生成：

```text
dataset/cut_data/
├── train/data.h5
├── val/data.h5
└── test/data.h5
```

H5 的 `data` 数据集形状为 `[切片数, 通道数, 切片长度]`，类型为 `float32`；`raw_labels`、`groups`、`source_files` 和 `slice_indices` 分别保存类别、投票分组、原始文件名和切片序号。默认使用读取较快的 LZF 压缩，可通过 `--h5_compression {none,lzf,gzip}` 修改，通过 `--h5_name` 修改文件名。多进程版本只在主进程写 H5，避免多个进程同时写一个 H5 导致文件损坏。

`overlap` 必须位于 `[0, 1)`；原始序列不足 `slice_len` 时会循环复制到目标长度。对于数十万小切片，单个 H5 通常明显快于逐个读取 CSV，因为它避免了大量文件打开、目录遍历和文本转浮点数的开销。

如需旧 CSV 格式，添加 `--output_format csv`：

```bash
python data_process/cut_data_multi.py --file_list ./dataset/split_data/train.txt --output_dir ./dataset/cut_data/train --slice_len 256 --workers 8 --output_format csv
```

CSV 模式的切片文件名采用 `<原始文件stem>_slice<四位切片序号>.csv`，并生成 `_index.csv`。训练时可直接从文件名恢复源文件分组。

## 3. 长度统计

```bash
python data_process/signal_length_stat.py --input_dir ./raw_data --output_dir ./data_process/stat_output
```

## 4. 数据集与归一化

训练代码调用 `load_cut_splits()`，推荐读取：

```text
data_root/
├── train/data.h5
├── val/data.h5
└── test/data.h5
```

也兼容旧的 `train/<类别>/*.csv`、`val/<类别>/*.csv`、`test/<类别>/*.csv`；同一 split 同时存在 H5 和 CSV 时优先读取 H5。默认特征列为 `CF_MHz`、`PW_us`、`PRI_us`，返回张量形状为 `[样本数, 通道数, 序列长度]`。`standard` 和 `minmax` 都按通道统计，且验证集、测试集复用训练集统计量。

当 H5 和 CSV 位于同一路径时，使用训练、预训练或推理命令的 `--data_format` 选择格式。`pretrain.py` 和 `inference.py` 默认使用 `auto`，`train.py` 默认使用 `csv`，建议显式指定：

```bash
--data_format auto   # 优先H5；pretrain/inference默认使用，train默认是csv
--data_format h5     # 强制H5，不存在时直接报错
--data_format csv    # 忽略H5，扫描类别目录下的CSV
```

独立推理使用 `load_cut_test()`，可以传入包含 `data.h5` 的测试目录，也可以直接传入 H5 文件路径。当前实现会整批载入每个 split，以减少 epoch 内的 I/O；大数据集需预留足够内存。

## 5. 数据增强

`RadarAugment` 只应用于训练数据。训练脚本示例：

```bash
python train.py --data_root ./dataset/cut_data --augment --augment_methods shift,gaussian_noise,scale,pulse_mask --augment_prob 0.5
```

可用方法为 `shift`、`gaussian_noise`、`scale`、`feature_mask`、`pulse_mask` 和 `reverse`。关闭增强使用 `--no-augment`。

## 6. 脉冲描述字可视化

指定一个目录后启动网页查看器，程序会递归列出其各级子目录中的 CSV：

```bash
python data_process/visualize_pdw.py --input_dir ./dataset/cut_data
```

默认会自动打开 `http://127.0.0.1:8775/`。左侧先显示输入根目录下的类别文件夹，点击文件夹后才懒加载该目录的子文件夹和 CSV，适合比较同类与异类样本，也能处理几十万级 CSV。选择 CSV 后，网页会绘制载频、脉宽和 PRI 随脉冲顺序变化的三张联动散点图；三张图共享 X 轴视图，在任一图中缩放或平移都会同步另外两张，且 Y 轴固定不变。二维图的 X/Y 轴可分别选择载频、脉宽或 PRI，因此可以查看脉宽-载频、脉宽-PRI、载频-PRI 等组合。Plotly 工具栏支持缩放、平移、复位和导出 PNG。端口可用 `--port` 修改，`--no_browser` 可禁止自动打开浏览器。

可指定端口，或禁止自动打开浏览器：

```bash
python data_process/visualize_pdw.py --input_dir ./raw_data --port 9000 --no_browser
```

CSV 读取由 `csv_features.py` 统一处理：有表头时会按 `CF_MHz/RF`、`PW_us/PW`、`PRI_us/PRI` 以及常见中英文别名识别，列顺序可以任意；无表头时无法从纯数字可靠推断物理含义，必须显式传入实际顺序，例如 `--no-header-order PRI,RF,PW`。如果不传且文件无可识别表头，程序会报错而不会静默交换特征。网页端使用 `https://cdn.plot.ly/plotly-2.35.2.min.js`，Python 端使用项目已有的 pandas；浏览器无法访问该 CDN 时页面可能能打开但无法绘图。

需要解释三项特征物理含义的切片、预训练、训练、推理、统计分析、RF 修正推理和可视化脚本支持 `--no-header-order`；`signal_length_stat.py` 只统计行数，不读取特征含义，因此没有该参数。H5 文件继续使用写入时保存的 `feature_cols` 元数据。例如：

```bash
python data_process/cut_data_multi.py --input_dir ./raw_data --file_list "" --no-header-order PRI,RF,PW
python train.py --data_format csv --no-header-order PRI,RF,PW
python inference.py --data_format csv --no-header-order PRI,RF,PW
```

## 7. 各型号 PDW 特征统计分析

原始 CSV 按型号文件夹组织时，可运行：

```bash
python data_process/analyze_pdw_features.py --input_dir ./raw_data --output_dir ./data_process/pdw_analysis
```

考虑到原始数据可能包含约 80 万个 CSV，默认每个型号随机分析最多 10 个文件，并行使用 4 个读取线程。需要全量精确统计时使用：

```bash
python data_process/analyze_pdw_features.py --input_dir ./raw_data --max_files_per_class 0 --workers 4
```

也可以只比较部分型号：

```bash
python data_process/analyze_pdw_features.py --input_dir ./raw_data --classes 5,6,7 --max_files_per_class 2000
```

主要输出包括：

- `class_feature_statistics.csv`：每型号、每特征的文件数、有效点数、均值、标准差、极值和抽样分位数。
- `pairwise_distribution_distance.csv`：型号两两之间三项特征的 Jensen-Shannon 分布距离，越接近 1 表示分布差异越大。
- `pdw_sequence_comparison.png`：类似连续雷达采样点示例的三项特征曲线，型号之间用虚线分隔。
- `feature_distribution_heatmaps.png`：各型号三项特征的概率分布热力图。
- `feature_boxplots.png`：各型号三项特征的抽样箱线图。
- `distribution_distance_heatmap.png`：型号间综合分布差异热力图。
- `read_errors.csv`：无法解析或没有有效数值的 CSV 清单。

`mean/std/min/max` 基于实际分析文件中的全部有效行计算；分位数、箱线图和分布图使用每型号最多 `--sample_size` 个均匀抽样点，从而控制内存占用。机械硬盘并行读取过多可能反而变慢，建议将 `--workers` 设为 1 或 2；SSD 可从 4 开始测试。

`signal_length_stat.py` 默认递归扫描 CSV，但当前实现最多处理排序后的前 1001 个文件（索引 0 到 1000），适合快速估计长度分布，不代表全量统计；同时会跳过 `_index.csv`。
