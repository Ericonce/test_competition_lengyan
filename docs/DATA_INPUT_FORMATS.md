# 比赛数据先转换为类别文件夹，再训练

训练和验证统一使用现有类别文件夹：`0/*.csv`、`1/*.csv`……，每个 CSV 是一条完整的脉冲序列。主办方提供其他组织方式时，先运行转换工具；三份 `configs/data/*.yaml` 模板仅用于描述转换前的数据。

## 1. 选择转换命令

先修改对应模板的 `root`，指向原始数据目录，再在仓库根目录执行下面对应的一条命令。`--output` 指定新的输出目录，可直接修改路径。

平铺 CSV，标签单独保存：

```shell
python tools/convert_dataset.py --data configs/data/label_file.yaml --output datasets/competition_converted
```

类别写在文件名里：

```shell
python tools/convert_dataset.py --data configs/data/filename.yaml --output datasets/competition_converted
```

多个样本合在一张大表里：

```shell
python tools/convert_dataset.py --data configs/data/table.yaml --output datasets/competition_converted
```

三种方式都会实际生成：

```text
datasets/competition_converted/
  data.yaml
  conversion_manifest.csv
  conversion_summary.json
  0/
    ...csv
  1/
    ...csv
  ...
```

原始文件保留。平铺样本按标签复制到对应类别目录，保留文件名和原文件字节；大表按完整样本拆成 `类别/0.csv`、`类别/1.csv` 等文件。输出目录必须尚不存在，并与原始数据目录分开；再次转换时换一个新的输出目录。转换失败会清理本次临时输出，修正源数据后可重新运行。

## 2. 训练时只指向转换后的 data.yaml

修改现有训练入口中的参数：

```python
model.train(
    # 原入口的 pipeline、模型、设备等参数继续保留。
    data="datasets/competition_converted/data.yaml",
    num_classes=100,
    dataset={"sample_ratio": 1.0, "train_ratio": 0.8, "val_ratio": 0.2,
             "seed": 2026, "stratified": True},
)
```

删除旧调用中的 `data_input={...}`。训练时传入转换前的模板或旧的非类别目录选项，会提示先转换。已有类别目录数据可继续直接训练。

转换只整理文件和标签，保留完整样本、脉冲顺序及原始物理值。采样、训练/验证划分、裁剪/补长、归一化沿用原来的训练流程。很多现有入口显式设置了 `sample_ratio=0.01`；需要全部数据时改成 `1.0`。

## 3. 三种原始数据如何配置

模板中的 `root` 默认相对于仓库，也可填写绝对路径。自己把描述 YAML 放进原始数据目录时，可用 `root: .` 和 `root_relative_to_yaml: true`。其他输入路径都相对于 `root`，必须位于其中。

### 平铺 CSV 加标签表

每个样本 CSV 保持原有 CF/PW/PRI 特征列。`labels.csv` 示例：

```csv
filename,label
a.csv,17
b.csv,42
```

编辑 `configs/data/label_file.yaml`：`labels_file` 指定标签文件，`files` 指定样本目录（默认 `.`）。标签中的文件名保持 `a.csv` 形式。`filename_column`、`label_column` 指定标签表的列名；顺序任意，但文件与标签必须一一对应，重复、缺失或多余标签会报错。

支持 TSV（`delimiter: "\t"`）和无表头的两列 TXT（`header: false`、`delimiter: whitespace`），TXT 每行为 `a.csv 17`。有空格的文件名请使用 CSV/TSV 标签表。这里的分隔符和编码描述标签表；独立样本 CSV 沿用现有逗号分隔、UTF-8 编码。

### 类别写在文件名里

例如 `class17_sample002.csv`，编辑 `configs/data/filename.yaml`：

```yaml
input:
  format: filename
  filename_regex: 'class(?P<label>\d+)_.*\.csv'
```

命名组 `(?P<label>...)` 提取类别；正则必须完整匹配文件名。文件为 `17_002.csv` 时改成 `'(?P<label>\d+)_\d+\.csv'`。`files` 指定平铺样本目录，`pattern` 默认 `*.csv`。无法匹配的文件会报错。

### 多个样本合在大表中

目前支持每行一个脉冲、有明确样本 ID 的 CSV/TSV 长表：

```csv
sample_id,label,pulse_index,CF_MHz,PW_us,PRI_us
a,17,0,3500,12,500
a,17,1,3501,12.2,501
b,42,0,4000,10,600
b,42,1,4001,10.1,601
```

编辑 `configs/data/table.yaml`：

```yaml
input:
  format: table
  table_file: samples.csv
  sample_id_columns: [sample_id]
  label_column: label
  order_column: pulse_index
```

- 同一样本的行可以交错。指定 `order_column` 时，按其数值升序整理脉冲；留空时保留文件中的出现顺序。
- 类别间复用样本编号时，用 `sample_id_columns: [label, sample_id]`。同一样本存在冲突标签、空编号或重复脉冲序号会报错。
- 每个样本导出一个 CSV，只写 `features` 指定的特征列，支持原有中文/英文列名别名。标签和样本 ID 写入转换清单，不进入训练特征。
- 表格需要表头；`delimiter: "\t"` 支持 TSV，`encoding: gb18030` 等可指定源表编码。
- 根据明确的样本 ID 拆分，不会按固定行数切段。其他大表布局（如每行是一条展开序列、没有样本编号）需要先按实际字段整理成上述长表。

## 4. 类别编号与转换核验

默认类别编号为 `0～num_classes-1`。原始标签为 `1～100` 时，在对应模板的 `input` 中设置 `label_offset: -1`。文本类别可设置：

```yaml
label_map: {radar_A: 0, radar_B: 1}
```

`label_map` 的值就是最终目录编号，不再叠加偏移。转换摘要保存原始配置及映射，提交预测时可据此恢复官方类别编号。

`conversion_manifest.csv` 记录每条原始样本、目标类别、输出文件及脉冲数；`conversion_summary.json` 记录总样本数和各类别数量。工具用训练读取器逐个核验输出 CSV，拒绝空序列、缺少特征、非数字或 NaN/Inf 数据。转换后的数据目录可以整体移动，训练无需依赖原始标签表或大表。

每次转换处理一个有标签数据池。主办方给出了官方训练/验证划分时，应分别转换并保留各自边界，再沿用项目已有的显式 train/val 目录或清单配置；不要混合成总池重新随机划分。独立无标签测试集继续使用现有文件夹推理入口，无需转换成有标签类别目录。
