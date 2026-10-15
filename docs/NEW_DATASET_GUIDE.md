# 接入一个新数据集

新数据集只要整理成 CF/PW/PRI 三通道 CSV，就可以使用现有模型训练和推理。建立数据目录和 data.yaml 后，直接修改现有 train/infer 入口的调用参数，沿用原 pipeline 和模型 YAML，无需复制配置或新增入口。下面以 **5 种辐射源**的新数据集 `new_radar` 为例。

比赛操作顺序：整理 CSV → 创建 data.yaml → 固定采样和训练/验证划分 → 关闭增强比较模型 → 在同一划分上比较增强方法及组合 → 各入选成员全量从头训练 → 用各成员最终权重投票推理。所有阶段复用现有入口、pipeline 和模型 YAML。已有数据无需经过合成生成器。

## 1. 整理训练/验证总数据池

在项目内创建以下目录，目录和文件名使用英文或数字：

```text
datasets/
  new_radar/
    data.yaml
    0/
      0.csv
      1.csv
    1/
      0.csv
      1.csv
    2/
      0.csv
    3/
      0.csv
    4/
      0.csv
```

每个数字文件夹代表一个具体辐射源类别；文件夹名称就是标签。5 类使用 `0～4`，原始类别名称或不连续编号需要先映射到这些编号，并自行保存编号与辐射源名称的对应关系。修改 num_classes 不会自动重编号或筛选类别。

每个 CSV 存放一条完整脉冲序列，每行是一个脉冲，第一行是表头：

```csv
CF_MHz,PW_us,PRI_us
3500.0,12.0,500.0
3510.0,12.2,502.0
3495.0,11.8,498.0
```

- `CF_MHz`：载频，单位 MHz。
- `PW_us`：脉宽，单位微秒。
- `PRI_us`：脉冲重复周期，单位微秒；不是 TOA 到达时间。

保留脉冲的时间先后顺序，PRI 数值本身不要求递增。序列长度可以不同，训练/推理加载时再补长或裁剪。CSV 须至少有一行有效数据，特征须为有限数值，不能含空值、NaN 或无穷大。建议统一使用上述列名和 UTF-8 编码。训练标签来自类别文件夹，不能将标签作为第四个输入通道。

CSV 文件名不要求从 0 连续编号，也可使用 `sample_001.csv`；普通数据会先建立文件名索引，选择好序列后才读取 CSV 内容。总池中只放用于训练/验证的数据，独立测试集另放一个目录。

## 2. 创建数据集描述文件（data.yaml）

在 `datasets/new_radar/data.yaml` 中填写：

```yaml
name: new_radar
root: .
root_relative_to_yaml: true
num_classes: 5
features:
  - CF_MHz
  - PW_us
  - PRI_us
cache: true
all: .
```

| 字段 | 作用 |
| --- | --- |
| `name` | 数据集标识，不参与拼接数据路径。 |
| `root` | 数据根目录，`.` 表示相对基准目录本身。 |
| `root_relative_to_yaml` | 为 true 时，root 相对于本 data.yaml 所在目录解析。 |
| `num_classes` | 辐射源类别数；5 类对应标签 0～4。 |
| `features` | 要读取的 CSV 列名及通道顺序；当前模型使用 CF/PW/PRI 三通道。 |
| `cache` | 是否缓存加载并调整长度后的训练/验证数组，减少重复读取 CSV。 |
| `all` | 总数据池相对于 root 的位置；`.` 表示类别文件夹直接位于 root 下。 |

这个配置的路径关系是：

```text
data.yaml 所在目录 = datasets/new_radar/
root = datasets/new_radar/
all  = datasets/new_radar/
样本 = datasets/new_radar/<标签>/<文件名>.csv
```

如果类别文件夹放在 `datasets/new_radar/sequences/` 下，保持 root 不变，将 `all` 改为 `sequences` 即可。保留相对路径后，整体迁移数据目录无需写死盘符或服务器路径。

缓存位于数据根目录的 `.pulsecls_cache/`。`cache: false` 关闭训练/验证数组缓存；小型文件名索引仍可能保存于该目录。当前训练入口显式设置了 `data_cfg={"cache": True, ...}`，要关闭缓存还需将调用中的 cache 改为 False，或删除该覆盖项。

创建 data.yaml 后，可以先运行 `python tools/visualize_dataset.py --data datasets/new_radar/data.yaml`，检查原始序列、类别数量、长度分布和特征重叠，再开始训练。可调参数和输出说明见 [数据集可视化](DATA_VISUALIZATION.md)。

## 3. 沿用原 pipeline，在训练调用中覆盖参数

选择现有 `train/` 下所需模型。例如 CNN 继续使用 `train/cnn_softmax_synthetic/train.py` 和 `configs/pipeline/cnn_softmax_synthetic.yaml`；pipeline 和模型结构 YAML 不需要复制或修改。

在 `model.train(...)` 中将 `data` 改为新 data.yaml，检查 `num_classes`、结果目录、采样比例和长度设置即可。显式的 `data` 会在加载配置时替换原引用，因此无需原合成数据集的 data.yaml 存在。完整调用范例见下一节。

如果类别数和其他设置相同，只改 `data` 路径即可；若入口仍显式写着 `num_classes=100`，新任务只有 5 类，则必须同步改为 5。各方法继续沿用原有网络、损失和适用的 AMP 设置。

固定长度、短序列补长和长序列裁剪策略的含义：

| 设置 | 含义 |
| --- | --- |
| `sequence_length: 256` | 每条输入统一为 256 个脉冲。 |
| `short: zero` | 不足长度时在末尾补零。 |
| `short: repeat` | 不足长度时循环重复原序列，直到达到指定长度。 |
| `long: head` | 超长时保留前面的指定数量脉冲。 |
| `long: center` | 超长时保留中间的指定数量脉冲。 |

采样和划分按 CSV 序列条数计算：先选总池的 sample_ratio，再把选中的序列按 train_ratio/val_ratio 分开。例如 1000 条序列、sample_ratio=0.5、train_ratio=0.8、val_ratio=0.2，会得到 400 条训练和 100 条验证。两项划分比例之和必须为 1。模型与增强筛选可先固定采样 10%，样本少时提高比例以覆盖各类训练/验证样本；只有最终全量训练才统一设置 sample_ratio=1.0、train_ratio=1.0、val_ratio=0.0。

## 4. 修改训练入口，先筛选模型再筛选增强

直接打开现有 `train/cnn_softmax_synthetic/train.py`，保留导入和 PROJECT_ROOT 初始化，在 main() 中修改 model.train()。以下展示主要参数；其他已列出的训练参数可以保留：

```python
model.train(
    pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
    data=PROJECT_ROOT / "datasets/new_radar/data.yaml",
    output_dir=PROJECT_ROOT / "results/cnn_softmax_new_radar",
    num_classes=5,
    epochs=50,
    batch_size=64,
    lr=0.001,
    device="auto",
    resume=False,
    resume_checkpoint=PROJECT_ROOT / "results/cnn_softmax_new_radar/last_model.pt",
    sequence_length=256,
    length_policy={"short": "zero", "long": "head"},
    dataset={
        "sample_ratio": 0.1,
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "seed": 2026,
        "stratified": True,
    },
    data_cfg={"cache": True, "progress": True},
    augment={
        "preset": PROJECT_ROOT / "configs/augment/default.yaml",
        "enabled": False,
    },
)
```

显式调用参数优先于 pipeline/data.yaml。只修改 YAML，却保留调用中的旧路径、num_classes=100 或 sample_ratio=0.01，会继续使用调用中的设置。要让某项完全由 YAML 控制，删除对应调用参数；num_classes=None 先继承 pipeline，pipeline 为 null 时训练才读取 data.yaml 的类别数。

路径直接写在调用中，不需要定义 PIPELINE_YAML、DATA_YAML 等常量，也不需要新建入口。新数据集默认从头训练，使用 resume=False 和独立 output_dir；仅继续同一类别任务、同一模型的已有训练时再开启 resume，epochs 此时表示额外轮数。

开启训练增强时将 augment.enabled 改为 True，具体方法、开关、强度和顺序编辑其 preset 指向的 YAML。也可以复制共享增强 YAML，为新数据集设置独立版本，避免同时改变其他实验的增强参数。

从项目根目录运行：

```bash
python train/cnn_softmax_synthetic/train.py
```

终端会显示设备、实际类别数、总序列数、采样数和训练/验证数。检查这些数量是否符合预期，再根据验证 Macro-F1 调参。

第一阶段关闭增强，分别运行候选模型的现有 train.py；固定数据池、dataset 字典、标签映射和长度策略，核对 dataset_selection.json 中的实际划分，保留后续参与投票的候选模型。

第二阶段针对每个候选模型，沿用同一采样和划分，依次比较无增强、单种增强、不同强度和多种增强串联。调用中开启 augment.enabled，增强 YAML 中设置方法开关、阈值和 order。每次从头训练，使用独立结果目录，并保留 configs/augment.yaml 快照；不同模型可以使用不同的最终增强方案。

按验证结果确定各成员的增强、训练参数、轮数及主模型，官方测试集不参与这些筛选。完整分阶段范例和投票入口代码见[主 README 的比赛流程](../README.md)。

## 5. 准备独立推理/测试文件夹

推理输入直接指定平铺 CSV 文件夹，不为测试输入创建 data.yaml，不放数字类别子目录：

```text
datasets/new_radar_test/
  0.csv
  1.csv
  2.csv
  labels.txt
```

CSV 的列名、单位和类别编号含义与训练一致。需要评估时准备 labels.txt，每行写文件名和标签，用空格或 Tab 分隔：

```text
0.csv 2
1.csv 0
2.csv 4
```

标签文件按文件名匹配，不依赖行顺序；必须恰好覆盖输入文件夹中的所有 CSV，不能重复、遗漏或多出文件名，标签必须位于 0～4。不需要评估时可不准备标签文件。用于报告测试性能的数据应与训练/验证总池独立。

## 6. 指定新权重进行推理

直接修改现有 `infer/cnn_softmax_synthetic/infer.py`，保留入口初始化和原 pipeline，显式传入新 data.yaml、测试目录及新权重：

```python
model.infer(
    pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
    data=PROJECT_ROOT / "datasets/new_radar/data.yaml",
    source=PROJECT_ROOT / "datasets/new_radar_test",
    checkpoint=PROJECT_ROOT / "results/cnn_softmax_new_radar/best_model.pt",
    save_dir=PROJECT_ROOT / "results/cnn_softmax_new_radar/infer_test",
    labels_file=PROJECT_ROOT / "datasets/new_radar_test/labels.txt",
    num_classes=5,
    device="auto",
    sequence_length=256,
    length_policy={"short": "zero", "long": "head"},
    inference={"batch_size": 256, "progress": True},
    augment={"preset": PROJECT_ROOT / "configs/augment/default.yaml"},
)
```

只预测时把 labels_file 改为 `""`，即使文件夹中有 labels.txt 也不读取标签；None 会继承 pipeline 的标签设置。推理不执行随机增强，增强 YAML 仅作为配置引用。推理按批读取 CSV，使用新权重中的训练归一化统计。

显式传入的 data.yaml，以及原 pipeline 引用的模型 YAML 和增强 YAML 仍需存在以解析配置；不需要原合成数据集的 data.yaml。文件夹推理不会读取训练 CSV，待预测数据由 source 指定。

```bash
python infer/cnn_softmax_synthetic/infer.py
```

结果保存在 save_dir 下的 predictions.csv；指定标签后还会生成 metrics.json、分类报告和混淆矩阵，包含 Macro-F1。投票时修改 infer_vote.py 中的类别数、测试目录和各成员 checkpoint；成员继续使用原 pipeline，在每个成员字典中增加 data 字段指向新 data.yaml。所有成员须在相同辐射源标签体系上训练。

## 7. 筛选结束后，各成员全量训练并投票推理

在训练调用中设置：

```python
dataset={"sample_ratio": 1.0, "train_ratio": 1.0, "val_ratio": 0.0, "seed": 2026, "stratified": True}
```

对所有入选投票成员分别设置上述 dataset，保留各自选定的增强方案及其他参数，使用 resume=False 和独立结果目录从头训练。轮数由前面的验证阶段确定。全量只包含可用的带标签训练池，不包含官方测试集。

这会关闭验证、基于验证的早停和调度，按 epochs 完成训练。每个成员的正式推理 checkpoint 改为其全量训练目录的 **final_model.pt**；采样筛选阶段的 best_model.pt 用于验证选方案，last_model.pt 用于断点续训。

正式测试使用现有 infer/infer_vote.py，设置比赛测试 source、统一类别数、主模型和各成员原 pipeline、比赛 data.yaml、最终 checkpoint。只启用已全量训练的成员，输出投票 predictions.csv，再按官方提交模板转换。无需复制投票配置或创建新推理入口。

## 8. 如果新数据集也由项目生成器生成

复制生成 YAML 后设置 output_dir、test_output_dir、num_emitters、每类序列数量、长度和噪声参数，分别执行训练总池和配套测试集生成脚本：

```bash
python tools/generate_synthetic_radar.py --config configs/generator/synthetic_radar_new.yaml
python tools/generate_synthetic_radar_test.py --config configs/generator/synthetic_radar_new.yaml
```

上面的 synthetic_radar_new.yaml 是需要自行复制并修改的示例文件名。生成器会自动创建训练总池的 data.yaml；随后按本指南修改训练/推理路径、类别数和采样参数。测试生成器复用该训练池的辐射源参数与配置，用独立随机流生成序列。真实数据或其他来源的数据不需要生成器的 emitters.json、summary.json 等元信息。

## 常见情况与处理

| 情况 | 应检查或修改的位置 |
| --- | --- |
| 找不到 data.yaml 或样本 | train() 的 data 路径、data.yaml 的 root/all、数字类别目录。 |
| 新数据只有 5 类，但训练仍显示 100 | 将调用中的 num_classes 改为 5；代码优先于 YAML。 |
| 实际只使用很少的数据 | 查看调用中的 dataset.sample_ratio，小数据集可改为 1.0。 |
| 报标签超出类别数 | 类别目录应是 0～K-1；减少 num_classes 不会自动删除其他类别。 |
| CSV 缺少特征或加载失败 | 检查表头、单位、有限数值和非空序列；第三通道应是 PRI。 |
| 修改增强 YAML 后没有增强 | 调用中的 augment.enabled 需为 True，并在 YAML 开启方法、设置顺序。 |
| 推理找不到权重 | 明确填写新训练目录的 best_model.pt；无验证训练使用 final_model.pt。 |
| 推理类别数与权重不同 | 以显式调用参数/推理配置为准；减少仅保留前 N 类，增加的未训练类别概率为 0，不会学习新辐射源。新任务应训练对应权重。 |
| labels.txt 报匹配错误 | TXT 文件名需与平铺测试目录的全部 CSV 一一对应，标签含义与训练一致。 |

更多设置见 [训练参数](TRAIN_CONFIGURATION.md)、[采样与划分](DATASET_SELECTION.md)、[增强方法](DATA_AUGMENTATION.md) 和 [推理与投票](INFERENCE_AND_TEST_DATA.md)。
