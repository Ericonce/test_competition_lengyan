# 辐射源脉冲序列分类：项目结构与比赛使用指南

本项目用于**根据一条脉冲序列，判断它属于哪个辐射源类别**。输入为三个通道：载频（CF）、脉宽（PW）、脉冲重复周期（PRI）。支持不同分类模型的训练、验证、单模型推理和多模型投票。

Windows 10 工作站可使用独立 Conda 环境进行 CPU 调试、训练和推理，安装与 VS Code 断点操作见 [Windows 调试指南](docs/WINDOWS_DEBUG.md)。

**参加比赛时使用主办方提供的数据，直接修改现有 train/infer 入口的参数即可。** 不需要复制 pipeline、模型 YAML 或另建训练入口，也不需要运行合成数据生成器。现有路径中的 `_synthetic` 是历史命名，不限制模型只能使用合成数据；`model.train(data=...)` 指向哪里，就读取哪个数据集。

训练数据统一按类别分文件夹；标签表、文件名标签或大表数据先用 [转换工具](docs/DATA_INPUT_FORMATS.md) 整理，再将 `data` 指向转换后的 `data.yaml`。

## 1. 项目结构：各部分负责什么

```text
competition/
├── README.md                         # 项目总览、比赛操作流程
├── train/
│   └── <方法名>_synthetic/train.py    # 各方法的独立训练入口
├── infer/
│   ├── <方法名>_synthetic/infer.py    # 各方法的独立推理入口
│   ├── infer_cnn_softmax_synthetic.py # 单模型推理范例
│   └── infer_vote.py                 # 多模型投票入口
├── configs/
│   ├── pipeline/                     # 实验配置：数据、网络、增强、训练/推理参数
│   ├── model/                        # 网络结构配置
│   ├── augment/                      # 增强方法的开关、执行顺序、阈值
│   ├── inference/                    # 投票等推理配置
│   └── generator/                    # 合成生成参数；接入比赛数据不需要它
├── pulsecls/                         # 项目核心实现
│   ├── cfg/                          # 默认参数、YAML 加载、参数覆盖和检查
│   ├── engine/                       # 训练、验证、推理、投票的公共流程
│   ├── models/classify/              # PulseCLS 分类接口，关联训练器和推理器
│   ├── nn/
│   │   ├── tasks.py                  # 根据 model 配置选择、创建网络
│   │   ├── classifiers/              # 分类模型实现、模块组合与 forward 数据流
│   │   ├── backbones/                # 骨干网络
│   │   ├── modules/                  # 可复用网络模块
│   │   ├── heads/                    # 特征投影等头部模块
│   │   └── hypothesis/               # 信号假设相关分支
│   ├── data/                         # CSV 读取、索引、采样划分、增强、合成生成器
│   ├── signals/                      # 部分方法使用的信号假设构造
│   ├── losses/                       # 分类与对比学习损失
│   └── utils/                        # 设备、指标、进度显示、结果输出
├── datasets/                         # 本地数据与各数据集自己的 data.yaml
├── results/                          # 建议保存权重、日志、指标、图片和推理 CSV
├── docs/                             # 各功能的中文详细文档
├── tools/
│   ├── convert_dataset.py            # 标签表/文件名标签/大表 → 类别文件夹
│   ├── clean_csv.py                  # 缺失值、非法脉冲修复，可选 IQR 异常检测
│   ├── slice_csv.py                  # 定长滑窗切片、补长、记录原 CSV 来源
│   ├── visualize_dataset.py          # 离线查看序列、特征分布和类别统计
│   ├── generate_synthetic_radar*.py   # 生成开发测试数据
│   └── ...                          # Windows 调试、NPU 检查等辅助工具
├── scripts/                          # 辅助处理、绘图、检查脚本；使用前检查适用数据
├── tests/                            # 自动化测试
└── requirements_npu.txt              # 昇腾环境依赖版本记录
```

支持 CNN、ResNet、TCN、InceptionTime、LSTM、BiLSTM、Transformer、Informer、MLP，以及 MambaSL 等方法。具体入口见[模型入口说明](docs/MODEL_ENTRYPOINTS.md)，结构来源与适配见[经典序列模型说明](docs/CLASSIC_SEQUENCE_MODELS.md)。

训练入口不直接拼接网络。实际调用关系是：

```text
train.py → PulseCLS.train() → 合并配置 → Trainer
        → nn/tasks.py 选择模型 → classifiers/ 中的模型类组装、执行网络

infer.py → PulseCLS.infer() → 合并配置 → Predictor
        → 加载训练权重及标准化统计量 → 逐批预测 → 保存 CSV
```

更详细的模块关系见[项目结构](docs/PROJECT_STRUCTURE.md)。

比赛数据相关功能可直接按下面的入口查找：

| 功能 | 入口 | 运行时机与结果 |
| --- | --- | --- |
| 格式转换 | `tools/convert_dataset.py` | 收到其他格式的带标签数据时运行，输出 `类别/*.csv` 和 `data.yaml`。 |
| 数据清洗 | `tools/clean_csv.py` | 切片之前运行，修复坏脉冲，输出新 CSV 和逐文件清洗记录。 |
| 定长切片 | `tools/slice_csv.py` | 原始 CSV 长度不同时运行，输出定长 CSV 和来源清单。 |
| 数据可视化 | `tools/visualize_dataset.py` | 检查样本、长度和分布，输出离线图片、统计表与本地汇总页。 |
| 特征归一化 | `model.train(normalization=...)`；实现位于 `pulsecls/utils/normalization.py` | 训练内部执行，支持六种方案；权重保存参数，推理自动复用。 |

转换、清洗、切片的具体命令见 **3.1**；采样划分和归一化设置见 **3.2**。清洗和切片保留物理单位，归一化通过训练参数选择，无需另外生成一份归一化 CSV。

## 2. 参数在哪里设置

| 文件或位置 | 负责的内容 |
| --- | --- |
| 数据集目录下的 `data.yaml` | 数据根目录、类别数、CSV 特征列、数据入口、缓存开关 |
| `configs/pipeline/*.yaml` | 组织一次实验：引用 data/model/augment，设置长度、采样划分、训练和推理参数 |
| `configs/model/*.yaml` | 所选模型的结构参数，例如隐藏维度、层数 |
| `configs/augment/*.yaml` | 各增强方法的开关、阈值及串联顺序 |
| `model.train(...)` / `model.infer(...)` | 在入口中显式设置常用参数，覆盖 YAML 的同名配置 |
| `results/<实验>/configs/` | 本次运行合并后实际生效的配置快照，便于核对与复现 |

**调用时显式传入的参数优先于 YAML。** 例如入口写了 `epochs=50`，只把 pipeline 改成 100 不会生效；需要修改入口，或删除该参数让它继承 YAML。嵌套字典按字段合并，未覆盖的字段继续继承配置。

训练和推理共用的输入长度、补长与裁剪策略放在 pipeline 顶层的 `sequence_length`、`length_policy`，也可以在调用中覆盖。数据路径建议使用项目相对路径，入口通过 `PROJECT_ROOT / "..."` 定位，方便迁移服务器。

## 3. 比赛流程总览

1. **拿到数据，确认格式**：将官方测试集单独保存；整理类别目录、CF/PW/PRI 表头与单位，保存标签编号映射。
2. **清洗，再按需切片**：训练池和测试集分别清洗；训练池滑窗切成模型需要的长度，保留切片与原 CSV 的对应关系。
3. **设置采样和划分，先选模型**：在训练入口填写 `dataset`；启动时自动采样并划分训练/验证，关闭增强比较候选模型。
4. **再选增强方法**：固定同一批数据和划分，比较无增强、单种增强和串联组合，确定每个投票成员的参数。
5. **各成员全量训练**：设置 `sample_ratio=1.0`、`train_ratio=1.0`、`val_ratio=0.0`，按选定轮数分别从头训练。
6. **推理并提交**：先核对单模型结果，再用各成员的 `final_model.pt` 投票，按官方模板整理提交文件。

```text
官方带标签训练池
  → 整理格式 → 清洗 → 滑窗切片
  → 训练启动时固定采样、按原 CSV 分组划分训练/验证
  → 关闭增强，比较模型 → 保留候选模型
  → 固定同一划分，比较增强方法和串联组合
  → 确定各投票成员的模型、增强、训练参数及主模型
  → 各成员分别在全部带标签数据上从头训练
官方测试集 → 同样清洗 → 各成员 final_model.pt → 多模型投票 → 按官方模板提交
```

所有阶段复用现有 train/infer 入口和原 pipeline、模型 YAML，在调用中修改参数即可。模型筛选、增强筛选和全量训练需要手动分阶段运行，当前入口不会自动搜索或批量完成这些步骤。

以下以 **CNN + Softmax、100 个类别、长度 256**为范例，直接修改现有 train/infer 入口。先激活运行环境（见第 4 节），下面命令都在仓库根目录执行。路径、类别数和长度按实际数据修改。

### 3.1 拿到数据：整理、清洗、切片

带标签的训练数据整理为：

```text
datasets/
├── competition_radar/
│   ├── data.yaml
│   ├── 0/
│   │   ├── 0.csv                     # 类别 0 的第一条序列
│   │   └── 1.csv
│   ├── 1/
│   │   └── 0.csv
│   ├── ...
│   └── 99/
└── competition_radar_test/            # 官方测试集：CSV 直接放在此目录
    ├── sample_0001.csv
    └── sample_0002.csv
```

每个 CSV 是一条序列，每行是一个脉冲，带表头，例如：

```csv
CF_MHz,PW_us,PRI_us
5200.0,12.0,800.0
5210.0,12.1,802.0
5195.0,11.9,799.0
```

- 三个通道依次为载频、脉宽、PRI，范例单位分别为 MHz、μs、μs。保持训练、验证、测试的列含义和单位一致。
- 第三通道是 **PRI，不是 TOA**。如果官方数据是 TOA、其他单位或其他存储格式，应先按照官方数据定义转换；本项目不会自动识别任意比赛原始格式。
- 类别目录使用 `0` 到 `K-1` 的整数。若官方标签是名称或其他编号，先映射成内部整数，并单独保存映射表，提交时反向还原。映射表需要自行维护。
- 序列原始长度可以不同，加载时统一补长或裁剪；模型看到的输入为 `[批大小, 3, 固定长度]`。
- 保留测试文件与官方样本 ID 的对应关系，避免转换格式后无法生成提交文件。

**① 格式转换：`tools/convert_dataset.py`。** 已经是上面的类别目录时跳过转换。其他三种组织方式按下面对应的一种处理，输出都使用 `datasets/competition_radar`；各模板的 `num_classes`、`features` 也要与实际数据一致。

**A. 所有样本平铺，标签单独保存。** 例如原始目录中有 `a.csv`、`b.csv` 和下面的 `labels.csv`：

```csv
filename,label
a.csv,17
b.csv,42
```

在现有 `configs/data/label_file.yaml` 中修改这些字段，其余字段保留：

```yaml
root: datasets/competition_raw
input:
  format: label_file
  files: .
  labels_file: labels.csv
  filename_column: filename
  label_column: label
```

```bash
python tools/convert_dataset.py --data configs/data/label_file.yaml --output datasets/competition_radar
```

按文件名匹配标签并复制到类别目录，例如 `17/a.csv`。`files` 是样本所在子目录，`labels_file` 是标签表；两者都相对于 `root`。标签表顺序任意，缺失、重复或多余的标签会报错。

**B. 类别写在文件名里。** 例如 `class17_sample002.csv` 属于类别 17。在 `configs/data/filename.yaml` 中修改：

```yaml
root: datasets/competition_raw
input:
  format: filename
  files: .
  pattern: '*.csv'
  filename_regex: 'class(?P<label>\d+)_.*\.csv'
```

```bash
python tools/convert_dataset.py --data configs/data/filename.yaml --output datasets/competition_radar
```

`(?P<label>...)` 提取类别编号；文件命名为 `17_002.csv` 时，将正则改成 `'(?P<label>\d+)_\d+\.csv'`。复制后保留文件名，无法匹配的文件会报错。

**C. 多个样本合在一张大表中。** 目前支持每行一个脉冲、带样本 ID 和标签的长表，例如 `samples.csv`：

```csv
sample_id,label,pulse_index,CF_MHz,PW_us,PRI_us
a,17,0,3500,12,500
a,17,1,3501,12.2,501
b,42,0,4000,10,600
```

在 `configs/data/table.yaml` 中修改：

```yaml
root: datasets/competition_raw
input:
  format: table
  table_file: samples.csv
  sample_id_columns: [sample_id]
  label_column: label
  order_column: pulse_index
```

```bash
python tools/convert_dataset.py --data configs/data/table.yaml --output datasets/competition_radar
```

同一 `sample_id` 的行合成一个 CSV，并按 `pulse_index` 排序；`order_column: null` 时保留出现顺序。类别之间复用样本 ID 时设 `sample_id_columns: [label, sample_id]`。一行展开一整条序列、缺少样本 ID 等其他大表格式，需要先按字段整理。

三种转换均保留原始文件，输出目录须尚不存在，并位于原始数据目录之外。原标签为 1～100 时，在对应模板的 `input` 下加 `label_offset: -1`；文字标签可用 `label_map: {radar_A: 0, radar_B: 1}`。后续提交预测时按保存的映射还原官方标签。

| 转换输出 | 内容与用途 |
| --- | --- |
| `0/*.csv`、`1/*.csv` 等 | 实际供训练读取的类别目录；独立 CSV 按标签复制，大表按完整样本拆分。 |
| `data.yaml` | 转换后的数据入口，使用相对路径，可随数据目录移动。 |
| `conversion_manifest.csv` | `source`、`label`、`output_path`、`num_pulses`，核对原样本与新路径。 |
| `conversion_summary.json` | 原始输入配置、标签映射、样本数和各类别数量。 |

转换会校验 CSV，空文件、缺列、非数字或 NaN/Inf 会报错并撤销本次输出。若独立样本 CSV 已有坏值，先把样本单独放在 `root/samples`，清洗到 `root/clean_samples`，再把模板的 `input.files` 改为 `clean_samples`；标签表保留在 `root`，文件名保持对应。长表中的坏值需先按样本整理后再清洗。TSV、TXT 标签及其他字段见[完整格式说明](docs/DATA_INPUT_FORMATS.md)。

下文假设训练池已整理到 `datasets/competition_radar`。

在该目录的 `data.yaml` 写入（工具生成过时核对即可）：

```yaml
name: competition_radar
root: .
root_relative_to_yaml: true
num_classes: 100
features:
  - CF_MHz
  - PW_us
  - PRI_us
cache: true
all: .
```

`root: .` 表示本 YAML 所在目录，`all: .` 表示这里的类别文件夹组成训练总池。此时不用手工创建 train/val 文件夹；官方测试集另放目录，无需测试 `data.yaml`。

**② 数据清洗：`tools/clean_csv.py`。** 训练池、测试集分别处理。输入是类别文件夹或每个 CSV 为一个样本的平铺目录；放在切片之前执行。

```bash
python tools/clean_csv.py --source datasets/competition_radar --output datasets/competition_clean
python tools/clean_csv.py --source datasets/competition_radar_test --output datasets/competition_test_clean
```

| 参数 | 作用 |
| --- | --- |
| `--source` | 原始样本目录，保留不变；每个 CSV 对应一条完整序列。 |
| `--output` | 新的清洗结果目录，须尚不存在，并位于原目录之外。 |
| `--iqr` | 默认 `0`，关闭统计筛选；例如 `3` 启用逐 CSV 的 IQR 异常检测。 |
| `--no-progress` | 关闭终端进度条，可选。 |

默认检测空单元格、非数字文本、NaN/Inf、float32 溢出或下溢为零、CF/PW/PRI 非正值，以及 `PW >= PRI`。**任意一个特征损坏时，把这个脉冲的三个特征整行一起修复**，文件名、类别目录、脉冲数量和顺序保留。

修复使用同一 CSV 内前后有效脉冲做线性插值。例如：

| 脉冲位置 | 原载频/脉宽/PRI | 清洗后 |
| --- | --- | --- |
| 1 | `3000,10,100` | 保持不变 |
| 2 | `缺失,11,105` | `3100,12,110` |
| 3 | `3200,14,120` | 保持不变 |

连续坏行按位置等比例插值；开头、末尾用最近有效脉冲填充。只剩一个有效脉冲时用它填充坏行；没有有效脉冲、空文件或缺少特征列时会报错并撤销本次输出，不会跳过样本。

需要额外处理数值有限的尖峰时，在上述命令末尾加 `--iqr 3`。程序先排除非法脉冲，再对每个 CSV 的各特征计算 `IQR=Q75-Q25`；任一特征超出 `[Q25-3×IQR, Q75+3×IQR]` 时，将该脉冲标记后插值。少于 4 个有效脉冲时跳过统计筛选，IQR 为零时偏离该常量的值也会被标记。正常跳频、多峰或长尾分布可保持默认 `--iqr 0`，按同一验证集确认是否启用。

| 清洗输出 | 内容与用途 |
| --- | --- |
| 原文件名的 CSV | 统一输出 `CF_MHz,PW_us,PRI_us` 三列，保留物理单位，不保留额外列。 |
| `_cleaning/report.csv` | `file`、`rows`、`invalid_rows`、`outlier_rows`、`repaired_rows`，逐文件核对修改量。 |
| `_cleaning/summary.json` | 来源目录、处理文件数、总修复脉冲数、IQR 参数。 |
| `data.yaml` / `labels.txt` | 类别目录生成可移动的 YAML；平铺测试目录已有的 `labels.txt` 原样复制。 |

先检查修复量是否符合预期，再进入后续训练。补零后的切片不要再清洗，以免把填充零识别为坏脉冲。详见[清洗规则](docs/DATA_INPUT_FORMATS.md#6-清洗缺失值和异常脉冲)。

**③ 定长切片：`tools/slice_csv.py`。** 将几百、几千、几万个脉冲的原 CSV 转成统一长度的多个窗口。训练数据保持类别文件夹，平铺测试目录也支持切片：

```bash
python tools/slice_csv.py --source datasets/competition_clean --output datasets/competition_sliced --length 256 --overlap 0.5
```

| 参数 | 作用 |
| --- | --- |
| `--source` / `--output` | 完整原 CSV 目录 / 新的切片目录；输出须尚不存在，并位于原目录之外。 |
| `--length 256` | 每片的脉冲数，也是后续模型的 `sequence_length`。 |
| `--overlap 0.5` | 重叠比例，范围 `[0,1)`；`0` 无重叠，`0.5` 重叠 50%。 |
| `--short zero` | 默认补零；换成 `--short repeat` 时循环重复该片已有脉冲。 |

步长为 `length - int(length × overlap)`。上面的设置步长 128；400 个脉冲的 CSV 会生成 `[0,256)`、`[128,384)`、`[256,400)` 三片，最后一片保留 144 个真实脉冲，再补长到 256。覆盖末尾后停止；整条原 CSV 不足一片时也会保留并补长。

输出命名为 `原文件名__slice000000.csv`，不改变原始文件。训练目录同时生成 `data.yaml`，并自动启用按原文件分组划分；测试目录不生成训练 YAML。

| `_slicing/manifest.csv` 字段 | 含义 |
| --- | --- |
| `slice_csv` / `source_csv` | 切片相对路径 / 完整原 CSV 的相对路径。 |
| `slice_index` | 同一原 CSV 内的切片序号，从 0 开始。 |
| `start_row` / `stop_row` | 原序列中覆盖的 `[起始,终止)` 行号，从 0 开始、不含表头。 |
| `valid_length` / `source_length` | 本片真实脉冲数 / 原 CSV 总脉冲数，用于区分补长部分。 |
| `label` | 类别编号；无标签测试集为空。 |

`_slicing/summary.json` 保存原文件数、总切片数、长度、重叠比例、步长和补长方式。**同一原 CSV 的切片在训练/验证中始终归入同一组**，避免同一个样本的相似切片进入两侧。训练完成后也可用来源清单找到同一原 CSV 的全部切片，供结果汇总。详见[切片规则](docs/DATA_INPUT_FORMATS.md#5-将长短不一的-csv-切成固定长度)。

后文统一使用：

| 用途 | 路径 |
| --- | --- |
| 训练，以及单模型/投票成员的 `data` | `datasets/competition_sliced/data.yaml` |
| 官方测试推理的 `source` | `datasets/competition_test_clean` |

跳过清洗或切片时，把所有入口路径同步改为实际目录。示例测试集只清洗，推理时按长度策略裁剪/补长，每个原 CSV 输出一条预测；测试集若也滑窗，需自行按清单汇总成原 CSV 的一条预测后提交。

**④ 数据可视化：`tools/visualize_dataset.py`。** 查看清洗后的原始序列、CF/PW/PRI 分布、序列长度和类别样本量，输出静态图片和本地汇总页，可离线使用：

```bash
python tools/visualize_dataset.py --data datasets/competition_clean/data.yaml --output-dir results/competition_data_preview
```

| 常用参数 | 默认值与作用 |
| --- | --- |
| `--data` / `--output-dir` | 数据 YAML / 输出目录，本例结果在 `results/competition_data_preview`。 |
| `--classes 0 1 2` | 只分析指定类别；不传时分析所有可用类别。 |
| `--sequences-per-class` / `--max-sequences` | 默认每类抽 20 条、总计最多 2000 条，控制读入量。 |
| `--plot-classes` / `--examples-per-class` | 默认图中展示最多 8 类、每类 2 条代表序列。 |
| `--seed` / `--dpi` | 默认 2026 / 300，控制固定抽样和图片分辨率。 |

完成后用浏览器打开输出目录的 `index.html` 查看汇总，或直接看 PNG；`class_counts.csv` 记录类别数量，`sequence_stats.csv` 记录每条抽样序列的长度、均值、标准差、最小/最大值，`sampled_pulses.csv` 保存绘图抽样点，`analysis_summary.json` 保存抽样配置和汇总。统计使用抽中 CSV 的完整原始序列，分布图使用抽样脉冲；不会改写样本或代替训练划分。

### 3.2 设置入口参数、采样划分和归一化

例如使用 CNN 时，打开 `train/cnn_softmax_synthetic/train.py`，保留原来的 `pipeline=...cnn_softmax_synthetic.yaml`。该 pipeline 继续提供 CNN 的网络、损失和默认设置；显式传入的 `data` 会替换其中的数据 YAML 引用，加载时无需原合成数据集的 data.yaml 存在。

| `model.train(...)` 中的参数 | 接入比赛数据时如何处理 |
| --- | --- |
| `pipeline` | 沿用当前模型的原路径，无需复制或改名 |
| `data` | 本例为 `datasets/competition_sliced/data.yaml` |
| `num_classes` | 与比赛类别数一致；原入口显式写了 100，不会自动随 data.yaml 改变 |
| `output_dir` | 建议改为比赛实验目录，避免覆盖旧结果 |
| `dataset` | 控制采样和训练/验证划分；固定 seed，各模型使用相同划分 |
| `sequence_length`、`length_policy` | 设置固定长度、短序列补长和长序列裁剪策略 |
| `normalization` | 三个特征共用同一方案、各自统计参数；默认 `zscore`，可切换六种方案 |
| `epochs`、`batch_size`、`lr`、`labelsmooth` 等 | 按需要直接在训练调用中调整 |
| `resume`、`resume_checkpoint` | 新数据集从头训练用 `resume=False`；续训时才加载对应任务的检查点 |
| `augment` | 在调用中设置总开关、方法、阈值及顺序；未覆盖的字段继承增强 YAML |

**如果类别数及其他设置都适用，只改 `data` 路径就可以切换训练数据。** 如果类别数不同，必须同步修改显式的 `num_classes`，或设为 `None` 继承配置；pipeline 的训练类别数也为 null 时，才读取 data.yaml 的类别数。类别数设置不会筛选或重新编号标签；所有训练标签必须落在 `0..num_classes-1`。

**① 采样与划分。** 训练启动时自动执行，只记录清单，不复制出 train/val 目录。在 `model.train(...)` 中设置：

```python
dataset={"sample_ratio": 0.1, "train_ratio": 0.8, "val_ratio": 0.2,
         "seed": 2026, "stratified": True},
```

| 参数 | 含义 |
| --- | --- |
| `sample_ratio` | 从完整训练池选用多少原样本，`0.1` 为 10%，`1.0` 为全部。 |
| `train_ratio` / `val_ratio` | 对已选样本进行训练/验证划分，两者之和为 1。 |
| `seed` | 固定数据采样和划分；与模型训练的随机种子分开。 |
| `stratified` | `True` 按类别分层分配，尽量保持类别比例；样本过少时提高采样比例。 |

先选总池的 10%，再按 8:2 划分。100 类、每类 8000 个原 CSV 时，共选 80000 个，其中训练 64000、验证 16000。滑窗后按**原 CSV**计数，同一原 CSV 的全部切片始终在同一组，实际切片数会不同；记录见 `dataset_selection.json`。比较模型时保持上述参数一致。

上述设置适用于允许自行划分的训练池。官方固定划分或按采集批次、场景、实体分组时，应分别处理并保留边界；当前 `dataset` 会重新随机划分，需要先调整入口。测试数据始终单独保存。

长度策略中，`short` 可用 `zero`、`repeat`、`error`；`long` 可用 `head`、`center`、`error`。切片长度、训练和推理的 `sequence_length` 均设为 256。移动切片数据时，将 `data.yaml` 和 `_slicing/` 清单一起保留，维持原样本分组。

**② 特征归一化。** 这部分已集成在所有模型共用的训练器里，通过 `normalization` 参数选择，实现位于 `pulsecls/utils/normalization.py`。三个通道分别计算统计量，例如载频减载频均值、除以载频标准差，脉宽和 PRI 各用自己的参数。

| `normalization` | 计算方式 | 作用 |
| --- | --- | --- |
| `"zscore"` | `(x-mean)/std` | 默认方案，按训练均值和标准差标准化。 |
| `"minmax"` | `(x-min)/(max-min)` | 默认缩放到 `[0,1]`，可设置其他范围。 |
| `"maxabs"` | `x/max(abs(x))` | 按训练最大绝对值缩放，保留零点。 |
| `"robust"` | `(x-median)/(Q75-Q25)` | 用中位数和四分位距缩放，减少极端值对统计量的影响。 |
| `"log_zscore"` | 先做 `sign(x)*log(1+abs(x))`，再 Z-score | 先压缩数值跨度，再拟合均值和标准差。 |
| `"none"` | `x` | 保留原始数值，便于对照实验。 |

在现有训练入口的 `model.train(...)` 中修改这一项即可，例如选择最大最小归一化：

```python
normalization="minmax",
```

需要详细设置时，将同一项改成字典。下面两种写法选择一种：

```python
# Min-Max 缩放到 [-1,1]，并将超出训练范围的数值限制在此范围内。
normalization={"method": "minmax", "feature_range": [-1, 1], "clip": True},
```

```python
# Robust 改用第 10～90 百分位的间距。
normalization={"method": "robust", "quantile_range": [10, 90]},
```

`eps` 默认 `1e-6`，作为分母下限，防止恒定通道除零。Min-Max 的 `clip` 默认 `False`，此时验证/测试值超出训练最小最大值时，结果也可超出目标范围；开启后才限制范围。Robust 保留异常点，清洗脚本的 IQR 才负责检测并修复异常脉冲。

统计只使用本次采样、划分后的**训练集有效脉冲**，不使用验证集或测试集。配置 `short: zero` 时，补零行不参与统计，变换后保持为零；重叠窗口或循环补长中的重复脉冲会参与训练统计。训练先在物理数值上做增强，再应用固定归一化参数；验证和推理直接使用这些参数。

| 归一化输出 | 用途 |
| --- | --- |
| 结果目录的 `normalization.json`、`normalization.npz` | 查看实际采用的方案、偏移、缩放和训练统计量。 |
| 各模型权重中的 `normalization` 字段 | 单模型推理、投票成员及续训自动还原同一变换。 |

切换方案做对照时设 `resume=False`，使用独立 `output_dir` 重新训练；续训会恢复原权重的方案。原始 CSV 和缓存保持物理数值，无需另存一份归一化数据。全局默认值在 `pulsecls/cfg/default.yaml`，入口显式参数优先；完整字段见[归一化说明](docs/NORMALIZATION.md)。

### 3.3 第一阶段：固定采样，先选择模型

在现有 `train/cnn_softmax_synthetic/train.py` 中修改 `model.train(...)`。沿用原文件的导入和 `main()`，调用范例：

```python
model.train(
    pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
    data=PROJECT_ROOT / "datasets/competition_sliced/data.yaml",
    output_dir=PROJECT_ROOT / "results/competition/model_selection/cnn_softmax_noaug",
    num_classes=100,
    epochs=50,
    batch_size=64,
    lr=0.001,
    labelsmooth=0.0,
    visualize=True,
    optimizer="adamw",
    weight_decay=0.0001,
    early_stopping_patience=6,
    device="npu:0",  # Windows 调试时用 cpu。
    use_amp=True,
    amp_dtype="float16",
    num_workers=0,
    seed=2026,
    resume=False,
    sequence_length=256,
    length_policy={"short": "zero", "long": "head"},
    normalization="zscore",
    dataset={"sample_ratio": 0.1, "train_ratio": 0.8, "val_ratio": 0.2,
             "seed": 2026, "stratified": True},
    data_cfg={"cache": True, "progress": True},
    augment={"preset": PROJECT_ROOT / "configs/augment/default.yaml", "enabled": False},
)
```

在项目根目录执行：

```bash
python train/cnn_softmax_synthetic/train.py
```

第一次可先设置 `epochs=1`、减小 `sample_ratio` 跑通流程，再按计划训练。训练启动时先索引和划分，再读取所选 CSV；开启增强时先增强、后归一化，验证和推理只使用训练集统计量。

开跑时核对终端打印的设备、总样本数、采样数和训练/验证数量。现有入口的 `sample_ratio=0.01` 只使用总池的 1%；使用全部数据时改为 `1.0`。

在其他候选模型的现有 train.py 中设置相同的 data.yaml、`dataset` 字典、标签映射和长度策略，并保持 `augment.enabled=False`。相同数据池和固定采样/划分设置使各模型使用相同序列；不要每换一个模型就换采样种子。

每个实验使用独立 `output_dir`，例如 `results/competition/model_selection/cnn_softmax_noaug`、`results/competition/model_selection/resnet_softmax_noaug`。依据验证 Macro F1、训练/推理耗时及错误互补性保留 2～3 个候选，数量按结果调整。不同模型比较时，核对 `dataset_selection.json` 中的实际划分一致。

当前最佳权重按**验证集 Macro F1**选择。若比赛使用其他评分规则，需要按官方指标评价方案，并相应调整选权重逻辑。

### 3.4 第二阶段：固定候选模型，再选择增强方法

保持第一阶段的采样比例、数据划分种子及训练/验证比例不变，针对每个入选模型比较：

| 实验 | 调整方式 |
| --- | --- |
| 无增强基线 | `model.train(..., augment={..., "enabled": False})` |
| 单种增强 | 训练调用总开关设为 True，增强 YAML 中只开启一个方法 |
| 同一方法不同强度 | 调整该方法的 SNR、触发概率或其他阈值，每次使用独立结果目录 |
| 多种增强串联 | 开启多个方法，用增强 YAML 的 `order` 设置先后顺序 |

继续复用 `configs/augment/default.yaml`，无需复制 pipeline 或模型 YAML。增强方法、阈值和顺序既可在 YAML 中设置，也可直接在现有 train.py 的 `augment` 字典中填写，代码中的同名字段优先，未填写项继承 YAML。

全部训练入口只保留一个 jittering 配置范例，可直接在 methods 中修改或添加其他方法；代码中的同名字段优先，方法开关为 False 可关闭 YAML 中已开启的同名方法。**总开关默认 `augment.enabled=False`，即使 train.py 或 YAML 中的方法为 True，也不执行任何增强。** 总开关改为 True 后，按合并配置中的方法开关执行。

执行顺序默认使用 YAML 的 `order`，也可取消入口中 order 范例的注释来覆盖；覆盖后的列表须包含所有已开启的方法。若选择在 YAML 中调节，下面是需要修改的字段：

```yaml
enabled: true
order:
  - jittering
  - time_shift
methods:
  jittering:
    enabled: true
    prob: 1.0
    snr_db: 40.0
  time_shift:
    enabled: true
    prob: 1.0
    ratio: 0.1
```

这是现有增强 YAML 的字段示例，其他方法设置可保留，但其开关须关闭；如果仍有其他方法开启，也必须在 `order` 中列出。上述串联表示先加噪，再对加噪结果做循环平移，不是分别增强后拼接。40 dB 和 0.1 是可调示例，不代表已在比赛数据上选出的最优值。

每次实验从头训练，使用例如 `results/competition/augmentation/cnn_softmax_jitter_shift` 的独立输出目录。模型结构和其他训练设置先保持一致，通过同一验证集比较增强带来的变化；验证集本身不做随机增强。

**不同模型可以选用不同增强方案，也可以保留无增强。** 记录每个成员最终采用的方法、顺序、阈值和训练轮数；运行结果的 `configs/augment.yaml` 会保存实际增强配置。共享增强 YAML 改成另一个成员的方案前，应确认当前运行已结束并保留前次快照；也可在各模型 train.py 的 `augment` 字典中覆盖成员专属设置，代码优先。

确定最终成员和主模型时，使用这批固定验证数据上的结果；可按样本路径对齐各实验的 `val_predictions.csv`，比较投票组合。官方测试集不参与模型或增强筛选。各增强参数的含义见[数据增强说明](docs/DATA_AUGMENTATION.md)。

### 3.5 查看各阶段结果与断点续训

| 输出文件 | 用途 |
| --- | --- |
| `best_model.pt` | 有验证集时，验证 Macro F1 最佳的权重 |
| `last_model.pt` | 最后完成轮次的检查点，用于断点续训 |
| `final_model.pt` | 无验证集时，训练完成后的最终权重 |
| `training_history.csv` | 各 epoch 的训练/验证准确率、Macro F1 等 |
| `final_metrics.json`、`training.log` | 最终指标、参数和运行日志 |
| `configs/`、`resolved_pipeline.json` | 实际生效配置及运行信息 |
| `dataset_selection.json` | 本次采样与划分记录 |
| `normalization.npz` | 训练集标准化统计量 |
| `figures/` | 开启可视化时的训练曲线及验证集分析图 |
| `val_*.csv/json`、`tsne/` | 验证集预测、每类指标、混淆矩阵和可视化特征等 |

`visualize=False` 可关闭绘图和 t-SNE，权重、日志、数值指标仍保存。有验证集且开启可视化时，训练结束会在验证集上输出每类指标图、混淆矩阵、t-SNE。完整文件说明见[训练结果输出](docs/TRAINING_OUTPUTS.md)。

断点续训时，在 `model.train(...)` 中把 `resume` 改为 `True`，将 `resume_checkpoint` 指向该实验的 `last_model.pt`，并设置本次额外训练的轮数。例如原来已经完成 30 轮，续训时 `epochs=20` 表示再训练 20 轮。

恢复优化器状态后，实际学习率可能来自检查点；以日志中的生效设置为准。详见[断点续训](docs/RESUME_TRAINING.md)。

### 3.6 第三阶段：每个投票成员分别全量训练

模型和增强方案确定后，按比赛规则使用全部可用的带标签训练数据。对每个入选模型，分别修改其现有 train.py 的调用：

| 参数 | 正式全量训练设置 |
| --- | --- |
| `output_dir` | `PROJECT_ROOT / "results/cnn_softmax_competition_final"` |
| `epochs` | 根据此前验证实验确定，例如 50 |
| `resume` | `False`，从头训练 |
| `dataset.sample_ratio` | `1.0` |
| `dataset.train_ratio` | `1.0` |
| `dataset.val_ratio` | `0.0` |

表中的 `dataset.xxx` 指训练调用中 `dataset={...}` 对应的字段。保留该成员第二阶段选定的增强方法、顺序、阈值及其他训练参数；第一阶段关闭增强不代表最终阶段也关闭增强。再次执行该模型原有训练入口即可。

所有参与投票的成员都需要完成这一步，并使用独立结果目录，例如：

| 投票成员（示例，实际以筛选结果为准） | 全量训练结果目录 | 推理权重 |
| --- | --- | --- |
| CNN + Softmax | `results/cnn_softmax_competition_final` | 该目录下 `final_model.pt` |
| ResNet + Softmax | `results/resnet_softmax_competition_final` | 该目录下 `final_model.pt` |
| TCN | `results/tcn_competition_final` | 该目录下 `final_model.pt` |

各成员的正式训练轮数在前面的验证阶段确定。这里的全量是整个带标签训练池，不包含官方无标签测试集。不要直接用采样筛选阶段的权重代替全量训练权重。

该阶段使用全部带标签训练池。没有验证集时，按设置轮数训练，不根据验证指标早停或调度学习率，输出 `final_model.pt` 和 `last_model.pt`。训练曲线和训练集数值报告仍保存，不生成验证分析图和 t-SNE。

**正式投票时，各成员 checkpoint 都指向自己全量训练目录的 `final_model.pt`。** 采样筛选阶段的 `best_model.pt` 用于验证和选方案；`last_model.pt` 用于续训。

### 3.7 第四阶段：用全量训练权重进行投票推理

先修改现有 `infer/cnn_softmax_synthetic/infer.py` 的 `data`、`source`、`checkpoint`、类别数和设备，运行 `python infer/cnn_softmax_synthetic/infer.py` 核对单模型结果。

再修改 `infer/infer_vote.py` 中的 `model.infer_vote(...)`。沿用原文件的初始化和 `main()`；下面以 CNN、ResNet、TCN 为例，**只启用已经筛选并完成全量训练的成员**：

```python
model.infer_vote(
    config=PROJECT_ROOT / "configs/inference/vote.yaml",
    source=PROJECT_ROOT / "datasets/competition_test_clean",
    output_dir=PROJECT_ROOT / "results/competition/infer_vote",
    labels_file="",
    main_model="cnn_softmax",
    num_classes=100,
    device="npu:0",  # Windows 调试时用 cpu。
    batch_size=256,
    progress=True,
    sequence_length=256,
    length_policy={"short": "zero", "long": "head"},
    augment={"preset": PROJECT_ROOT / "configs/augment/default.yaml"},
    models=[
        {
            "name": "cnn_softmax",
            "enabled": True,
            "pipeline": PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
            "data": PROJECT_ROOT / "datasets/competition_sliced/data.yaml",
            "checkpoint": PROJECT_ROOT / "results/cnn_softmax_competition_final/final_model.pt",
        },
        {
            "name": "resnet_softmax",
            "enabled": True,
            "pipeline": PROJECT_ROOT / "configs/pipeline/resnet_softmax_synthetic.yaml",
            "data": PROJECT_ROOT / "datasets/competition_sliced/data.yaml",
            "checkpoint": PROJECT_ROOT / "results/resnet_softmax_competition_final/final_model.pt",
        },
        {
            "name": "tcn",
            "enabled": True,
            "pipeline": PROJECT_ROOT / "configs/pipeline/tcn_synthetic.yaml",
            "data": PROJECT_ROOT / "datasets/competition_sliced/data.yaml",
            "checkpoint": PROJECT_ROOT / "results/tcn_competition_final/final_model.pt",
        },
    ],
)
```

在项目根目录执行：

```bash
python infer/infer_vote.py
```

- `source` 本例为清洗后的官方测试 CSV 平铺文件夹，不需要测试 data.yaml；各样本文件名与原测试集一致。
- 每个成员的 `data` 指向比赛训练任务的描述文件，覆盖原 pipeline 中的合成数据引用；配置解析仍需要该 data.yaml、原模型 YAML 和增强 YAML 存在，但不会读取训练 CSV。
- `checkpoint` 指向该成员正式全量训练产生的 `final_model.pt`。关闭没有训练好的成员，不要混入旧任务权重。
- `main_model` 必须与某个已启用成员的 `name` 一致，根据前面验证阶段结果选择；范例中的 CNN 仅为占位选择。
- 所有成员的类别编号含义、特征定义和单位必须一致；各成员使用自己权重中的训练标准化统计量。此范例各模型使用一致的长度策略。
- 推理不执行随机数据增强；训练时的增强通过已经训练好的权重发挥作用。
- 每个开启的模型依次推理并显示进度，然后统计票数。只有一个成员开启时，结果等同于单模型预测。

投票选择票数最多的类别。最高票并列时，主模型预测若在并列候选中就优先选它；若主模型预测不在最高票候选中，则按当前实现选择其中编号最小的类别。所有模型预测都不同时，采用主模型预测。

官方测试集没有公开标签时，保持 `labels_file=""`，只推理。若对独立有标签数据进行评估，可填写标签 TXT 路径，每行 `文件名.csv 类别整数`。不应使用官方隐藏测试结果反复筛选方案。详细输出见[推理与投票说明](docs/INFERENCE_AND_TEST_DATA.md)。

训练和推理的 `data` 都指向本次比赛数据的 `data.yaml`。更换数据后同步修改推理入口及投票成员的 `data`，配置解析就无需原合成数据目录。通用 `infer/infer.py` 也支持 `--data 数据目录/data.yaml`。

### 3.8 将投票结果整理为比赛提交文件

上述范例的投票输出保存在：

```text
results/competition/infer_vote/predictions.csv
```

`prediction` 是最终投票类别，`<成员名>_prediction` 是各模型的预测；`vote_count`、`vote_fraction`、`tied` 等字段记录票数和并列情况。`confidence` 为获选类别的票数占比，并非校准后的分类概率。有标签评估时还包含真实标签及评估结果。

**该 CSV 是项目通用输出，不保证符合比赛提交格式。** 根据主办方模板生成最终提交文件：

1. 根据 `filename` 与官方样本 ID 建立对应关系。
2. 使用最终投票的 `prediction`，通过保存的标签映射表还原为官方标签。
3. 按官方模板的列名与样本顺序整理；不要只依赖文件名排序。
4. 核对每个官方测试样本恰好一条预测，无缺失、无重复，再保存为规定格式并提交。

## 4. 运行环境与设备

从项目根目录运行上述命令。核心依赖包括 PyTorch、NumPy、PyYAML、tqdm、scikit-learn、Matplotlib 等；使用昇腾时还需要匹配的 CANN、驱动和 `torch_npu` 环境。

使用自带双 Conda 镜像时，通过网页工作区把代码上传、解压到 `/workspace/test_competition_lengyan`。进入网页终端后，先修改训练入口的数据路径及参数，再执行：

```bash
tmux new -s radar
source /data/lym/anaconda3/etc/profile.d/conda.sh
conda activate radar
cd /workspace/test_competition_lengyan
python train/cnn_softmax_synthetic/train.py
```

`/data/lym/anaconda3` 随镜像携带，比赛宿主机无需有同名目录。按 `Ctrl+B`，松开再按 `D` 可分离；重新打开终端执行 `tmux attach -t radar` 返回。容器停止时任务也会停止。

也可用 `bash scripts/run_on_ascend.sh train/cnn_softmax_synthetic/train.py` 自动激活 `radar`，将运行缓存和日志放到项目的 `results/ascend_runtime`。推理时把最后的脚本路径换成对应的 `infer.py`。

`device="auto"` 优先选择 NVIDIA CUDA，其次昇腾 NPU，最后 CPU。可以显式设置 `cuda:0` 或 `npu:0`。`use_amp=True` 开启混合精度，范例通过 `amp_dtype="float16"` 指定精度类型。每个训练 epoch、验证阶段与推理批次均有终端进度提示。

比赛训练建议明确设置 `device="npu:0"`，第二张卡用 `npu:1`；设备不可用时会直接报错。Windows 调试使用 `device="cpu"`。两个单卡任务分别运行时，使用不同的 `output_dir`。

训练数据目录只读时，启用的缓存自动改存到系统临时目录；上述启动脚本会统一写入可写的 `results/ascend_runtime/cache/data`。`data_cfg={"cache": False}` 同时关闭样本缓存和文件索引缓存，无需修改原始数据目录权限。

`requirements_npu.txt` 是昇腾环境的依赖版本记录，包含平台相关包，应按目标服务器环境核对；CUDA 环境不应直接照搬其中的昇腾软件栈。昇腾 Transformer/TCN 适配与实测说明见[昇腾性能记录](docs/ASCEND_PERFORMANCE_20261015.md)。

归一化的六种方案、参数和输出文件见 **3.2**；它由训练器自动执行，切换数据和运行设备时仍沿用同一套接口。

数据集、权重和结果已被 `.gitignore` 忽略，从 GitHub 下载代码后需要自行准备比赛数据与可用权重。迁移机器时，检查所有自定义路径和运行环境。

## 5. 合成数据与辅助工具：开发测试时使用

合成数据功能用于无比赛数据时的开发、流程检查与模拟实验，**不是参加比赛的必要步骤，也不能替代官方数据评估**。

```bash
python tools/generate_synthetic_radar.py
python tools/generate_synthetic_radar_test.py
```

生成参数由 `configs/generator/synthetic_radar.yaml` 配置，生成入口显式引用该 YAML。现有 `_synthetic` 训练和推理入口初始指向这套数据，也可以按前文直接改为比赛数据。生成机制与重新生成方法见[合成数据引擎](docs/SYNTHETIC_RADAR.md)。

`tools/` 还包含数据可视化与 NPU 检查工具；`scripts/` 中部分脚本绑定合成实验或特定服务器数据规模，使用前应检查脚本路径和断言，不能直接视为比赛训练入口。

## 6. 详细文档

- [完整中文文档目录](docs/README.md)
- [项目结构与模型组装](docs/PROJECT_STRUCTURE.md)
- [新数据集接入与 data.yaml 字段](docs/NEW_DATASET_GUIDE.md)
- [将标签表、文件名标签、大表转换为类别目录后训练](docs/DATA_INPUT_FORMATS.md)
- [训练参数与配置覆盖](docs/TRAIN_CONFIGURATION.md)
- [采样与训练/验证划分](docs/DATASET_SELECTION.md)
- [六种特征归一化方案与参数](docs/NORMALIZATION.md)
- [训练输出文件、指标与图片](docs/TRAINING_OUTPUTS.md)
- [数据增强方法、阈值与顺序](docs/DATA_AUGMENTATION.md)
- [数据集可视化工具](docs/DATA_VISUALIZATION.md)
- [比赛测试数据校验与分类整理](docs/OVERLAP_POSTCHECK.md)
- [各模型的训练与推理入口](docs/MODEL_ENTRYPOINTS.md)
- [独立测试集推理与多模型投票](docs/INFERENCE_AND_TEST_DATA.md)
- [断点续训](docs/RESUME_TRAINING.md)

代码测试可在已安装相应依赖的环境中运行：

```bash
python -m pytest -q tests
```
