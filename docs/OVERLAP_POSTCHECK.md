# 测试 CSV 重叠后检验

入口为 `tools/postcheck_overlap.py`。指定训练总池的 `data.yaml`、平铺测试文件夹和一个新的输出目录，即可检查每个测试 CSV 是否与某条已知类别的训练序列相同或存在完整连续包含关系。确认类别唯一的文件按 `classified/<类别编号>/<原文件名>.csv` 原样复制，原始训练和测试 CSV 保留。

## 环境与依赖

本机验证使用 Python 3.10.20，代码沿用仓库现有 Python 语法和 API。核验计算使用 CPU，不创建神经网络或加载权重。脚本复用 `pulsecls` 的读取与进度模块，导入时会继承仓库现有 PyTorch、pandas、scikit-learn 等依赖，应使用已经能正常 `import pulsecls` 的项目环境。

本工具新增的第三方依赖为 **0**，没有调整 `requirements_npu.txt` 或项目全局配置：

| 使用内容 | 来源与仓库版本记录 |
| --- | --- |
| NumPy | 已有依赖，`requirements_npu.txt` 为 `numpy==1.26.4`；原始特征、字节序和行去重 |
| PyYAML | 已有依赖，`PyYAML==6.0.2`；读取配置和导出 data.yaml |
| tqdm | 已有依赖，`tqdm==4.67.1`；通过现有 `Progress` 显示进度 |
| SQLite | Python 标准库 `sqlite3`，由解释器自带，无需安装 PyPI 包 |
| SHA、CSV、JSON、文件复制、临时目录 | Python 标准库 `hashlib/csv/json/shutil/tempfile` 等 |

环境导入检查可运行：

```bash
python -c "import pulsecls, numpy, yaml, tqdm, sqlite3; print(numpy.__version__, yaml.__version__, tqdm.__version__, sqlite3.sqlite_version)"
```

Windows CUDA 环境沿用项目自己的 CUDA 依赖，昇腾环境沿用匹配 CANN 的 PyTorch/torch_npu；`requirements_npu.txt` 包含昇腾专用包，不应直接照搬到 Windows。工具没有设置 OpenMP、CUDA 或 NPU 环境变量，也不更改全局包版本。

## 输入与运行

沿用仓库的数据规范：训练标签来自数字类别文件夹，特征顺序及列名来自 `data.yaml.features`。默认使用 `CF_MHz/PW_us/PRI_us` 三通道，支持现有读取函数的中文列名别名。测试输入只扫描 `source` 直接下级的所有 CSV，各类别混放，不需要测试标签或测试 data.yaml。子文件夹的 CSV 不参与检查，与现有文件夹推理入口一致。

```text
datasets/competition_train/
  data.yaml
  0/
    a.csv
  1/
    b.csv
datasets/competition_test/
  0.csv
  1.csv
  2.csv
```

训练 `data.yaml` 示例：

```yaml
name: competition_train
root: .
root_relative_to_yaml: true
all: .
num_classes: 100
features: [CF_MHz, PW_us, PRI_us]
cache: true
```

从项目根目录运行，Windows PowerShell 和 Linux 均可使用同一条命令：

```bash
python tools/postcheck_overlap.py --data datasets/competition_train/data.yaml --source datasets/competition_test --save-dir results/overlap_postcheck
```

也可以直接修改脚本 `main()` 中 `postcheck_overlap(...)` 的路径与参数，然后运行 `python tools/postcheck_overlap.py`。命令行显式参数优先。

| 参数 | 默认与含义 |
| --- | --- |
| `data` / `--data` | 训练总池 data.yaml；与现有训练入口一致，初始指向 `datasets/synthetic_radar_overlap/data.yaml` |
| `source` / `--source` | 测试 CSV 混放目录；初始为 `datasets/synthetic_radar_test` |
| `save_dir` / `--save-dir` | 新输出目录；初始为 `results/overlap_postcheck`，已存在时拒绝覆盖 |
| `selection_file` / `--selection-file` | 默认 None，核验 data.yaml 的完整数据清单；可指定一次训练保存的 `dataset_selection.json`，仅使用其中 `train` 清单 |
| `min_overlap_rows` / `--min-overlap-rows` | 默认 1，允许任意非空短序列的完整包含；可提高到 8、16 等，减少极短片段的赋值，整条相同匹配始终保留 |
| `progress` / `--no-progress` | 默认显示读取、核验、复制进度；命令行使用 `--no-progress` 关闭 |

默认核验 `data.yaml` 的整个数据池，其中可能含有训练/验证两部分；这是已知类别参考数据，不代表这些文件全部参与过某次模型训练。若只想核验某次训练实际使用的序列，运行：

```bash
python tools/postcheck_overlap.py --data datasets/competition_train/data.yaml --source datasets/competition_test --selection-file results/cnn_competition/dataset_selection.json --save-dir results/overlap_train_only
```

`dataset_selection.json` 的相对样本路径按当前 `data.yaml.root` 解析，可以迁移机器后复用。默认数据清单通过现有 `load_inventory()` 读取，兼容 `all` 指向目录或样本 JSON 清单的形式。目录索引缓存仍按仓库规则保存在训练数据根目录的 `.pulsecls_cache/`。

## 匹配规则

| `rule` | 含义 |
| --- | --- |
| `sha256_equal` | 训练 CSV 与测试 CSV 的完整文件 SHA-256 相同，并核实所选特征的完整序列相同 |
| `sequence_equal` | 所选特征的完整序列数值和顺序相同；允许 BOM、换行、表头别名、列顺序、额外非特征列和小数写法不同 |
| `test_in_train` | 整条测试序列是某条训练序列的一个连续片段，可位于开头、中间或结尾 |
| `train_in_test` | 整条训练序列是某条测试序列的一个连续片段，可位于开头、中间或结尾 |

每行完整的特征向量需要相同，脉冲顺序需要相同。读取保留原始 float64 数值和全长，正负零按同一个数处理。不会先裁剪、补长、相对 TOA 平移、增强或标准化。数值相同以 CSV 解析后的 float64 为准，不使用 float32 舍入、容差、均值相似度或近邻距离来赋标签。

只共享部分脉冲，但双方都没有完整包含另一方；跳着取行；行顺序打乱；加噪声后的近似片段，均不满足这些规则。只比较 `data.yaml.features` 声明的输入特征，额外列不参与数值校验。

任一规则成立即可提供一个候选类别。全部匹配训练文件都属于同一类时，赋予该类标签。同一测试文件匹配多个类别时，记录 `conflict`，保留全部证据，等待人工处理；即便其中存在 SHA 完全相同的文件，也不会覆盖其他类别的冲突证据。多个同类训练文件匹配时只复制一次测试文件。

默认最短一行的完整包含也能提供标签；常量序列或很短的片段可能出现在多个类别中。使用 `min_overlap_rows` 可按业务需要提高包含所需长度，仍需查看冲突和匹配来源。

## 输出

```text
results/overlap_postcheck/
  classified/
    data.yaml
    0/
      0.csv
    1/
      2.csv
    ...
  assignments.csv
  matches.csv
  matched_labels.txt
  class_counts.csv
  conflicts.csv
  unmatched.csv
  reference_files.csv
  reference_data.yaml
  summary.json
```

- `classified/`：为 `0..num_classes-1` 各类建立文件夹，只复制唯一类别匹配的原始测试 CSV，保留文件名和文件字节。其 `data.yaml` 使用相对根目录和 `all: .`，可按仓库标准数据集读取；零匹配时没有可读取的样本。
- `assignments.csv`：所有测试文件各一行。`status` 为 `matched/conflict/unmatched`；`prediction` 只为唯一匹配填写整数标签，其他情况留空。还记录候选类别列表、匹配训练文件数、最强证据方式、原始长度、两个 SHA 和输出相对路径。
- `matches.csv`：每对匹配的训练文件和测试文件一行，包括训练路径、类别、规则、两边长度、完整短序列的脉冲数、两边文件与特征 SHA。多次重复出现时记录该文件对的第一个匹配位置；不截断同类或跨类别匹配证据。
- `matched_labels.txt`：`文件名.csv<Tab>类别编号`，与仓库标签文本的行格式一致，**只覆盖唯一匹配文件**。原始测试目录还有未匹配或冲突 CSV 时，此文件不能直接作为原目录的完整评估标签，现有 `read_labels()` 要求恰好覆盖全部输入。
- `class_counts.csv`：每类的参考训练文件数、唯一匹配测试文件数和涉及该类的冲突测试文件数。一条冲突文件可以计入多个类别。
- `conflicts.csv`、`unmatched.csv`：分别筛出冲突、未匹配测试文件，字段与 `assignments.csv` 相同。
- `reference_files.csv`：本次实际核验的全部参考 CSV 路径、标签、长度、文件 SHA-256 和特征 SHA-256。
- `reference_data.yaml`：输入 data.yaml 的原始快照；`summary.json` 记录路径、特征、核验范围、参数和统计。

`matches.csv` 的起始行编号从 0 开始，排除 CSV 表头。例如 `test_in_train` 且 `train_start_row=10`，表示测试整条序列对应训练数据第 11 个脉冲开始的连续片段。

`feature_sha256` 针对按所选列顺序排列、正负零归一后的 little-endian float64 完整序列字节计算；文件 SHA 针对原 CSV 文件字节计算。包含关系在候选筛选后再次做完整连续字节核验，首行相同本身不会构成匹配。

脚本使用临时 SQLite 索引保存测试序列和脉冲行索引，逐文件扫描参考训练序列。内存主要随单文件大小、一个测试文件的统计与候选数量变化。输出父目录需有足够磁盘空间保存临时索引和匹配 CSV，索引在完成或报错后自动清理。文件存在大量完全重复内容时，全部匹配关系本身可能很多，耗时及 `matches.csv` 大小也会增加。

输出目录必须位于训练根目录和测试目录外，且事先不存在；样本 JSON 清单指向外部 CSV 时，输出也必须位于其类别目录外。重新运行请换一个目录。全部核验和复制完成后才生成正式结果目录，缺列、空文件、非有限数值或输入变化会报错。可以将唯一匹配标签作为预测后的确定性补充，并通过清单保留其来源。匹配得到的标签应与独立评估的真实标签区分保存。

## 2026-10-16 本地规范复查记录

对照本地仓库提交 `c8b141b` 的 `tools/visualize_dataset.py`、`docs/PROJECT_STRUCTURE.md`、数据读取模块和依赖清单进行复查：

- 工具放在 `tools/`，测试放在 `tests/`，使用说明已加入主 README 和文档目录。
- 常用路径直接写在 `main()` 的函数调用中，CLI 显式参数优先；所有入口参数已添加中文注释。
- 复用 `load_yaml/remap_repo_path/load_inventory/read_sequence_csv/find_inference_csvs/Progress`；类别目录、列别名、数字文件名排序、CSV 编码及相对根目录规则与仓库一致。
- 已注释数值编码、脉冲行对齐、双向候选完整性、冲突策略、SQL 参数分批和 Windows 下临时索引清理。
- 补充了外部样本清单的实际类别目录检查，以及损坏训练清单的明确报错；主模型、训练、推理、全局配置及依赖清单保持原样。

验证环境为仓库目录外的独立环境 `../_delivery_build/postcheck_compat_env/`，继承本地 HSI 环境，覆盖安装以下仓库已有版本；未修改 HSI 或全局 Python 环境：

| 项目 | 验证值 |
| --- | --- |
| 系统与 Python | Windows，Python 3.10.20 |
| NumPy / pandas | 1.26.4 / 2.3.3，与仓库依赖记录一致 |
| PyYAML / tqdm | 6.0.2 / 4.67.1，与仓库依赖记录一致 |
| OpenCV | 4.11.0.86，与仓库记录一致；用于避免继承环境中 4.13 对 NumPy 2 的版本要求 |
| PyTorch | 2.10.0+cu128，继承本地环境；核验计算使用 CPU |
| SQLite / pytest | Python 自带 SQLite 3.52.0 / pytest 9.0.3 |

验证结果：

- 标准导入命令正常结束；未使用临时 PYTHONPATH 或 OpenMP 环境变量覆盖。
- `pip check` 返回 `No broken requirements found.`。
- AST 导入与仓库依赖清单比对：新增第三方依赖 0；`requirements_npu.txt` 内容一致。
- 后检验 27 项测试、仓库现有读取/索引/路径 8 项相关回归测试，共 35 项通过。空 CSV 反例触发现有读取器的一条预期警告，测试验证其正确报错。
- 独立 CLI 样例核验 2 个参考 CSV、5 个测试 CSV，得到唯一匹配 4、冲突 0、未匹配 1；分类输出与原 CSV 字节一致。
- `results/` 下的运行结果被现有 `.gitignore` 忽略，`git diff --check` 通过。

本机 HSI 环境原本缺少 pandas，且现有核心库版本与仓库昇腾记录不同，因此直接切换到 HSI 不等于使用了上述验证环境。默认合成数据目录未包含在代码仓库中，无参数运行前仍需准备该数据，或用 `--data/--source` 指向实际数据。上述 Windows 复查未运行全仓库测试。

后续已在真实 Ascend 910B3、PyTorch 2.1.0 / torch-npu 2.1.0.post10 容器中完成相同的 35 项测试、实际 NPU 运算检查，以及基于现有雷达 CSV 构造的 30 个完整命令行样例。环境、结果和容器既有依赖告警见 [真实昇腾验证记录](OVERLAP_POSTCHECK_ASCEND_20261016.md)。
