# 比赛测试数据校验

运行 `tools/postcheck_overlap.py`，检查比赛测试 CSV 与已知类别训练序列是否相同或存在完整连续包含关系。类别唯一的匹配文件按训练集的数字类别目录原样复制，并输出标签清单和每类数量统计。

## 输入与运行

训练数据沿用现有 `data.yaml`：标签来自数字类别文件夹，读取列及顺序由 `features` 指定。测试 CSV 的各类别混放在同一个文件夹中，只读取该目录直接下级的 CSV。

```text
datasets/competition_train/
  data.yaml
  0/a.csv
  1/b.csv
datasets/competition_test/
  0.csv
  1.csv
  2.csv
```

训练 `data.yaml` 示例，类别数按实际数据填写：

```yaml
name: competition_train
root: .
root_relative_to_yaml: true
all: .
num_classes: 100
features: [CF_MHz, PW_us, PRI_us]
cache: true
```

在项目根目录使用现有项目 Python 环境执行，Windows PowerShell 与 Linux 命令相同：

```bash
python tools/postcheck_overlap.py --data datasets/competition_train/data.yaml --source datasets/competition_test --save-dir results/overlap_postcheck
```

也可以直接修改脚本 `main()` 中的参数，再运行 `python tools/postcheck_overlap.py`。命令行显式参数优先。

| 参数 | 用法 |
| --- | --- |
| `--data` | 训练总池 data.yaml；默认 `datasets/synthetic_radar_overlap/data.yaml`，与现有训练入口一致 |
| `--source` | 平铺测试 CSV 文件夹；默认 `datasets/synthetic_radar_test` |
| `--save-dir` | 新结果目录；默认 `results/overlap_postcheck`，已存在时拒绝覆盖 |
| `--selection-file` | 可选，指定某次训练的 `dataset_selection.json`，只使用其 train 清单 |
| `--min-overlap-rows` | 包含匹配的最短脉冲数，默认 1；可提高为 8、16 等。整条相同匹配始终保留 |
| `--no-progress` | 关闭进度显示 |

默认参考 data.yaml 的整个已知类别总池。若只使用某次训练实际选中的文件，追加 `--selection-file results/<训练结果目录>/dataset_selection.json`。

使用现有 NumPy、PyYAML、tqdm 和 Python 标准库，新增第三方依赖为 0。匹配计算使用 CPU，沿用现有能正常导入 `pulsecls` 的项目环境。

## 匹配与赋值

| 规则 | 认定条件 |
| --- | --- |
| `sha256_equal` | 原始 CSV 文件 SHA-256 相同，且完整特征序列相同 |
| `sequence_equal` | 完整特征序列的数值和顺序相同，允许表头别名、列顺序、换行、小数写法等格式差异 |
| `test_in_train` | 整条测试序列连续出现在某条训练序列中 |
| `train_in_test` | 整条训练序列连续出现在某条测试序列中 |

比较 data.yaml 指定的完整原始特征，保留 float64 精度和脉冲顺序。包含关系要求整条短序列连续出现；跳行、打乱顺序和带数值误差的近似片段不计入匹配。

全部匹配来源只有一个类别时赋值，多个同类来源只复制一次。涉及多个类别时记为 `conflict`，保留匹配来源，标签留空。极短片段可能出现在多个类别，可通过 `--min-overlap-rows` 提高包含长度要求。

## 输出与比赛使用

```text
results/overlap_postcheck/
  classified/
    data.yaml
    0/<原测试文件名>.csv
    1/<原测试文件名>.csv
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

| 输出 | 比赛中的用途 |
| --- | --- |
| `classified/` | 唯一类别匹配的测试 CSV，保留文件名和内容；附带相对路径 data.yaml |
| `assignments.csv` | 每个测试文件一行；`status` 为 matched/conflict/unmatched，只有 matched 填写 `prediction` |
| `matches.csv` | 匹配训练文件、类别、规则和连续片段位置；行号从 0 开始，不含表头 |
| `matched_labels.txt` | `文件名.csv<Tab>类别编号`，只覆盖唯一匹配文件 |
| `class_counts.csv` | 每类参考文件数、唯一匹配数和涉及该类的冲突数量 |
| `conflicts.csv` / `unmatched.csv` | 分别列出类别冲突和未匹配文件，继续使用模型预测或人工处理 |
| `reference_files.csv` / `reference_data.yaml` / `summary.json` | 本次使用的数据清单、配置与数量汇总 |

用校验标签补充模型预测时，按 `filename` 对齐，只将 `status=matched` 的 `prediction` 覆盖到对应预测行。冲突和未匹配文件保留模型预测，最终按比赛模板的样本顺序和标签映射整理提交文件。本脚本输出分类目录与清单，预测合并及比赛模板转换由现有提交流程完成。

`matched_labels.txt` 只含部分文件，不能直接当作原始完整测试目录的评估标签。现有 `read_labels()` 要求标签恰好覆盖全部输入文件。

输出目录需事先不存在，并位于训练根目录、测试目录及外部参考 CSV 的类别目录之外。原始 CSV 保留，分类结果采用复制方式。临时索引在处理结束后自动清理，输出父目录需预留索引与分类文件所需磁盘空间。
