# 选择比赛数据的组织方式

训练和划分验证支持四种输入：原有 `class_folders`（类别文件夹），以及 `label_file`（平铺 CSV 加标签表）、`filename`（文件名包含类别）、`table`（大表包含多个样本）。三种新方式直接引用原始数据，不复制或重命名样本；大表按样本编号建立字节位置索引，不拆成几十万个小文件。

## 选择方式

修改任意现有训练入口的 `model.train(...)`，指定数据描述和输入选项：

```python
model.train(
    # 其他模型、设备等参数沿用原入口。
    data="你的数据目录/data.yaml",
    num_classes=100,
    data_input={"format": "label_file", "labels_file": "labels.csv"},
    dataset={"sample_ratio": 1.0, "train_ratio": 0.8, "val_ratio": 0.2,
             "seed": 2026, "stratified": True},
)
```

数据描述的共同部分如下；`root` 是原始文件所在的数据根目录：

```yaml
name: competition
root: .
root_relative_to_yaml: true
num_classes: 100
features: [CF_MHz, PW_us, PRI_us]
cache: true
input:
  format: label_file
  labels_file: labels.csv
```

`input` 放在 `data.yaml` 中。代码中的 `data_input={...}` 会整体替换这组输入选项；未提供该参数时使用数据 YAML。也支持 `data_cfg={"input": {...}}` 和 `**{"data_cfg.input.format": "filename"}`。省略 `input` 时保留原有类别文件夹读取。恢复原模式可传入 `data_input="class_folders"`。

仓库提供三个可编辑模板：`configs/data/label_file.yaml`、`configs/data/filename.yaml`、`configs/data/table.yaml`。先修改模板的 `root` 和对应字段，再把训练入口的 `data` 指向模板即可。其他参数继续受入口中显式参数控制，例如当前大多数入口的 `sample_ratio=0.01`；需要使用全部数据时改成 `1.0`。

## 平铺 CSV 加标签表：label_file

目录示例：

```text
competition/
  data.yaml
  a.csv
  b.csv
  labels.csv
```

每个 CSV 是一个脉冲序列，仍使用 CF/PW/PRI 特征列。`labels.csv` 示例：

```csv
filename,label
a.csv,17
b.csv,42
```

选择：

```python
data_input={
    "format": "label_file",
    "files": ".",
    "labels_file": "labels.csv",
    "filename_column": "filename",
    "label_column": "label",
}
```

`files` 和 `labels_file` 相对于数据根目录。CSV 放在 `sequences/` 时，设 `files="sequences"`；标签中的文件名保持 `a.csv` 形式。标签按文件名匹配，行顺序任意。所选 CSV 与标签须逐一对应，重复、缺失、多余文件名会报错；同目录的标签 CSV 自动排除。

支持 TSV 或无表头 TXT。例如 `labels.txt` 的每行是 `a.csv 17`：

```python
data_input={"format": "label_file", "labels_file": "labels.txt",
            "header": False, "delimiter": "whitespace"}
```

TSV 使用 `delimiter="\t"`。有表头时通过 `filename_column`、`label_column` 指定列名；无表头时严格采用“文件名、标签”两列。CSV 文件名可包含空格，有这类名称时使用 CSV/TSV 标签表。

## 类别写在文件名中：filename

例如 `class17_sample002.csv`、`class42_sample003.csv`：

```python
data_input={"format": "filename",
            "filename_regex": r"class(?P<label>\d+)_sample\d+\.csv"}
```

`filename_regex` 必须完整匹配文件名，使用命名组 `(?P<label>...)` 指出类别部分。默认规则是 `r"class(?P<label>\d+)_.*\.csv"`。例如名称为 `17_002.csv` 时，使用 `r"(?P<label>\d+)_\d+\.csv"`。读取器只使用明确提供的规则；无法匹配的文件会报错。

`files` 默认 `.`；`pattern` 默认 `*.csv`，可改为 `sample_*.csv` 等，以选择该平铺目录里的样本。`pattern` 使用文件名通配符，`filename_regex` 使用正则表达式，两者含义不同。

## 多个样本合在大表中：table

每行一个脉冲，样本 ID 标识整条序列：

```csv
sample_id,label,pulse_index,CF_MHz,PW_us,PRI_us
a,17,0,3500,12,500
a,17,1,3501,12.2,501
b,42,0,4000,10,600
b,42,1,4001,10.1,601
```

```python
data_input={"format": "table", "table_file": "samples.csv",
            "sample_id_columns": ["sample_id"], "label_column": "label",
            "order_column": "pulse_index"}
```

- 同一个样本的行可以交错分布，也可以连续排列。
- `order_column=None` 时，每条序列保持其在文件中的出现顺序。指定该列时，按其数值升序排列；同一序列的序号须为互不重复的有限数值。
- 每个样本 ID 对应一个固定类别。若不同类别复用了样本编号，使用 `sample_id_columns=["label", "sample_id"]` 等复合编号；未明确配置的标签冲突会报错。
- 特征列通过 `data.yaml` 的 `features` 选择，支持已有英文/中文列名别名；标签、样本编号和脉冲序号不会进入模型输入。
- 表格需要表头，支持 CSV 和 TSV（`delimiter="\t"`）。文本编码默认 `utf-8-sig`，可通过 `encoding` 指定，例如 `gb18030`。
- 第一遍扫描整张表，建立每个样本的行位置索引；后续只定位读取选中的序列。索引只保留 ID、标签和位置，不把整张特征大表载入内存。连续存放每个样本的全部行，通常比大量交错存放更节省索引空间。
- 输出清单中的 `samples.csv#sample=...` 是包含转义 ID 的样本标识，可区分同一大表里的不同序列。

表格必须提供明确的样本边界。每行已经代表一条展开序列、多个样本共用一个单元格、只有脉冲而无样本编号等其他布局，需要先转换为上述逐脉冲长表格式。不要按 256 行自动切段，以免跨样本拼接。

## 类别编号、划分、缓存和推理

默认标签为 `0～num_classes-1`。主办方使用 `1～100` 时设 `label_offset=-1`；使用文本类别名时配置：

```python
data_input={"format": "label_file", "labels_file": "labels.csv",
            "label_map": {"radar_A": 0, "radar_B": 1}}
```

配置 `label_map` 后，映射值就是最终类别编号，`label_offset` 不再叠加。映射必须明确覆盖输入类别；训练配置和权重保存这份映射。模型预测仍输出映射后的整数，提交时按主办方要求转换编号。不会根据类别出现顺序临时推断映射。

三种新模式都作为有标签的总数据池，继续按 `dataset.sample_ratio/train_ratio/val_ratio/seed/stratified` 采样和划分；划分单位为完整样本，训练和验证不共享样本。它们不读取标签表中的 `split` 列；若主办方已有官方划分，应明确配置并保留对应边界，避免直接把官方训练/验证合为这里的总池。大表模式也支持 `tools/visualize_dataset.py` 使用原始序列绘图。

输入索引和加载后的数组缓存位于数据根目录 `.pulsecls_cache/`。修改标签表、文件名规则、映射、数据文件或相关读取配置后，会重新建立对应索引/数组缓存。数组仍保留原始物理值；采样完成后沿用现有裁剪/补长、训练集拟合归一化、按批训练等流程。`dataset_selection.json` 记录输入选项、样本标识和实际训练/验证清单。

使用同一数据配置执行 `model.val(...)`，可以评估划分出的验证集。已有“独立 CSV 测试文件夹”推理方式保持可用；这些 `data_input` 选项用于训练/划分验证，不会把独立测试文件夹中的标签文件当成训练输入。推理归一化仍使用权重保存的参数。
