# 辐射源脉冲序列分类：项目结构与比赛使用指南

本项目用于**根据一条脉冲序列，判断它属于哪个辐射源类别**。输入为三个通道：载频（CF）、脉宽（PW）、脉冲重复周期（PRI）。支持不同分类模型的训练、验证、单模型推理和多模型投票。

**参加比赛时使用主办方提供的数据，直接修改现有 train/infer 入口的参数即可。** 不需要复制 pipeline、模型 YAML 或另建训练入口，也不需要运行合成数据生成器。现有路径中的 `_synthetic` 是历史命名，不限制模型只能使用合成数据；`model.train(data=...)` 指向哪里，就读取哪个数据集。

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
├── tools/                            # 数据生成、可视化、NPU 性能检查等工具
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

1. **整理比赛数据**：建立三通道 CSV 和 data.yaml，保存官方标签与内部编号的映射。
2. **从整个带标签训练池中固定采样**：先选取一部分完整序列，再将这部分数据划分为训练集和验证集。
3. **先选模型**：关闭数据增强，在相同采样、相同划分上比较候选模型，保留后续参与投票的模型。
4. **再选增强方法**：针对入选模型比较无增强、单种增强、多种增强串联，确定每个模型的增强方案和训练参数。
5. **全量训练投票成员**：将采样比例设为 1、训练比例设为 1、验证比例设为 0，分别从头训练所有入选模型，保留各自选定的增强方案。
6. **投票推理并提交**：在现有 infer_vote.py 中加载各成员全量训练的权重，对官方测试集投票，再按官方模板整理提交文件。

```text
官方带标签训练池
  → 固定采样一部分 → 划分训练/验证
  → 关闭增强，比较模型 → 保留候选模型
  → 固定同一划分，比较增强方法和串联组合
  → 确定各投票成员的模型、增强、训练参数及主模型
  → 各成员分别在全部带标签数据上从头训练
官方测试集 → 各成员 final_model.pt → 多模型投票 → 按官方模板提交
```

所有阶段复用现有 train/infer 入口和原 pipeline、模型 YAML，在调用中修改参数即可。模型筛选、增强筛选和全量训练需要手动分阶段运行，当前入口不会自动搜索或批量完成这些步骤。

以下以 **CNN + Softmax、5 个类别**为范例。直接使用现有 `train/cnn_softmax_synthetic/train.py` 和 `infer/cnn_softmax_synthetic/infer.py`。`competition_radar`、`competition_radar_test` 是示例数据目录名称，`results/*competition*` 是示例结果目录；类别数、输入长度等按实际比赛修改。

### 3.1 整理主办方数据

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
│   ├── 2/
│   ├── 3/
│   └── 4/
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

在 `datasets/competition_radar/data.yaml` 写入：

```yaml
name: competition_radar
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

这里的路径解析为：`data.yaml 所在目录 / root / all`，即 `datasets/competition_radar/`。程序在该目录下查找数字类别子目录。`features` 指定读取的列及顺序；`cache` 开启缓存。`all` 表示待采样、划分的带标签总数据池，不包含官方测试集。

`data.yaml` 只描述数据；内部训练/验证比例和使用总池的比例在 `model.train(dataset={...})` 中设置，覆盖 pipeline 默认值。官方测试集推理直接指定文件夹，不需要再创建测试 `data.yaml`。

可以先检查原始序列和分布：

```bash
python tools/visualize_dataset.py --data datasets/competition_radar/data.yaml --output-dir results/competition_data_preview
```

### 3.2 选择现有入口，直接修改调用参数

例如使用 CNN 时，打开 `train/cnn_softmax_synthetic/train.py`，保留原来的 `pipeline=...cnn_softmax_synthetic.yaml`。该 pipeline 继续提供 CNN 的网络、损失和默认设置；显式传入的 `data` 会替换其中的数据 YAML 引用，加载时无需原合成数据集的 data.yaml 存在。

| `model.train(...)` 中的参数 | 接入比赛数据时如何处理 |
| --- | --- |
| `pipeline` | 沿用当前模型的原路径，无需复制或改名 |
| `data` | 改为比赛数据集的 data.yaml 路径 |
| `num_classes` | 与比赛类别数一致；原入口显式写了 100，不会自动随 data.yaml 改变 |
| `output_dir` | 建议改为比赛实验目录，避免覆盖旧结果 |
| `dataset` | 检查采样、训练/验证比例；使用全部训练池时设 `sample_ratio=1.0` |
| `sequence_length`、`length_policy` | 设置固定长度、短序列补长和长序列裁剪策略 |
| `epochs`、`batch_size`、`lr`、`labelsmooth` 等 | 按需要直接在训练调用中调整 |
| `resume`、`resume_checkpoint` | 新数据集从头训练用 `resume=False`；续训时才加载对应任务的检查点 |
| `augment` | 在调用中设置总开关、方法、阈值及顺序；未覆盖的字段继承增强 YAML |

**如果类别数及其他设置都适用，只改 `data` 路径就可以切换训练数据。** 如果类别数不同，必须同步修改显式的 `num_classes`，或设为 `None` 继承配置；pipeline 的训练类别数也为 null 时，才读取 data.yaml 的类别数。类别数设置不会筛选或重新编号标签；所有训练标签必须落在 `0..num_classes-1`。

长度策略中，`short` 可用 `zero`（尾部补零）、`repeat`（循环重复）、`error`；`long` 可用 `head`（保留前段）、`center`（中心裁剪）、`error`。采样以完整 CSV 序列为单位，先采样再划分；`train_ratio + val_ratio` 须为 1。

### 3.3 第一阶段：固定采样，先选择模型

在现有 `train/cnn_softmax_synthetic/train.py` 中修改参数，例如下面的完整范例。示例先从总池采样 **10%**，再将这部分数据按 **80% / 20%** 划分；比例是可调示例，不是固定要求。pipeline 保持原路径，常用设置直接传给 `model.train(...)`：

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls import PulseCLS


def main():
    model = PulseCLS()
    return model.train(
        pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
        data=PROJECT_ROOT / "datasets/competition_radar/data.yaml",
        output_dir=PROJECT_ROOT / "results/competition/model_selection/cnn_softmax_noaug",
        num_classes=5,
        epochs=50,
        batch_size=64,
        lr=0.001,
        labelsmooth=0.0,
        visualize=True,
        optimizer="adamw",
        weight_decay=0.0001,
        early_stopping_patience=6,
        device="auto",
        use_amp=True,
        amp_dtype="float16",
        num_workers=0,
        seed=2026,
        resume=False,
        resume_checkpoint=PROJECT_ROOT / "results/competition/model_selection/cnn_softmax_noaug/last_model.pt",
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


if __name__ == "__main__":
    main()
```

在项目根目录执行：

```bash
python train/cnn_softmax_synthetic/train.py
```

流程是：**建立文件索引 → 选择要使用的序列 → 划分训练/验证 → 读取所选 CSV → 统一长度 → 训练**，不会先加载全部 CSV 内容再采样。训练开启增强时，先在原始特征上增强，再用训练集统计量按通道标准化；验证和推理不执行随机增强，使用训练时计算的统计量。

例如总池有 100,000 条序列，上述设置会选取 10,000 条，再分成约 8,000 条训练、2,000 条验证；其余 90,000 条不参与本阶段。采样规模应足够覆盖各类别的训练和验证样本，过少时应提高 `sample_ratio`。

在其他候选模型的现有 train.py 中设置相同的 data.yaml、`dataset` 字典、标签映射和长度策略，并保持 `augment.enabled=False`。相同数据池和固定采样/划分设置使各模型使用相同序列；不要每换一个模型就换采样种子。

每个实验使用独立 `output_dir`，例如 `results/competition/model_selection/cnn_softmax_noaug`、`results/competition/model_selection/resnet_softmax_noaug`。依据验证 Macro F1、训练/推理耗时及验证错误的互补性选择后续投票成员。可先保留 2～3 个候选，也可以按实际结果调整数量；不是只保留一个模型。

上述示例适用于从官方带标签训练池中自行划分验证集。**如果比赛规定了固定验证集，或要求按采集批次、场景、辐射源实体分组划分，应按该规则配置数据入口和划分；本例按类别随机划分不保证这些分组隔离。** 官方测试集始终单独用于推理。不同模型比较时，可核对 `dataset_selection.json` 中的实际样本划分。

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

直接修改现有 `infer/infer_vote.py`。保留其初始化方式，在 `model.infer_vote(...)` 中设置比赛测试目录、各成员权重及主模型，无需复制任何配置。下面以 CNN、ResNet、TCN 三个成员为例，**实际只启用已经筛选并完成全量训练的成员**：

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls import PulseCLS


def main():
    model = PulseCLS()
    return model.infer_vote(
        config=PROJECT_ROOT / "configs/inference/vote.yaml",
        source=PROJECT_ROOT / "datasets/competition_radar_test",
        output_dir=PROJECT_ROOT / "results/competition/infer_vote",
        labels_file="",
        main_model="cnn_softmax",
        num_classes=5,
        device="auto",
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
                "data": PROJECT_ROOT / "datasets/competition_radar/data.yaml",
                "checkpoint": PROJECT_ROOT / "results/cnn_softmax_competition_final/final_model.pt",
            },
            {
                "name": "resnet_softmax",
                "enabled": True,
                "pipeline": PROJECT_ROOT / "configs/pipeline/resnet_softmax_synthetic.yaml",
                "data": PROJECT_ROOT / "datasets/competition_radar/data.yaml",
                "checkpoint": PROJECT_ROOT / "results/resnet_softmax_competition_final/final_model.pt",
            },
            {
                "name": "tcn",
                "enabled": True,
                "pipeline": PROJECT_ROOT / "configs/pipeline/tcn_synthetic.yaml",
                "data": PROJECT_ROOT / "datasets/competition_radar/data.yaml",
                "checkpoint": PROJECT_ROOT / "results/tcn_competition_final/final_model.pt",
            },
        ],
    )


if __name__ == "__main__":
    main()
```

在项目根目录执行：

```bash
python infer/infer_vote.py
```

- `source` 是官方测试 CSV 的平铺文件夹，不需要测试 data.yaml。
- 每个成员的 `data` 指向比赛训练任务的描述文件，覆盖原 pipeline 中的合成数据引用；配置解析仍需要该 data.yaml、原模型 YAML 和增强 YAML 存在，但不会读取训练 CSV。
- `checkpoint` 指向该成员正式全量训练产生的 `final_model.pt`。关闭没有训练好的成员，不要混入旧任务权重。
- `main_model` 必须与某个已启用成员的 `name` 一致，根据前面验证阶段结果选择；范例中的 CNN 仅为占位选择。
- 所有成员的类别编号含义、特征定义和单位必须一致；各成员使用自己权重中的训练标准化统计量。此范例各模型使用一致的长度策略。
- 推理不执行随机数据增强；训练时的增强通过已经训练好的权重发挥作用。
- 每个开启的模型依次推理并显示进度，然后统计票数。只有一个成员开启时，结果等同于单模型预测。

投票选择票数最多的类别。最高票并列时，主模型预测若在并列候选中就优先选它；若主模型预测不在最高票候选中，则按当前实现选择其中编号最小的类别。所有模型预测都不同时，采用主模型预测。

官方测试集没有公开标签时，保持 `labels_file=""`，只推理。若对独立有标签数据进行评估，可填写标签 TXT 路径，每行 `文件名.csv 类别整数`。不应使用官方隐藏测试结果反复筛选方案。详细输出见[推理与投票说明](docs/INFERENCE_AND_TEST_DATA.md)。

如需单独检查某个成员，也可使用其现有 `infer/<方法名>_synthetic/infer.py`，修改 data、source、checkpoint 和类别数；正式提交流程使用上述投票入口。

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

`device="auto"` 优先选择 NVIDIA CUDA，其次昇腾 NPU，最后 CPU。可以显式设置 `cuda:0` 或 `npu:0`。`use_amp=True` 开启混合精度，范例通过 `amp_dtype="float16"` 指定精度类型。每个训练 epoch、验证阶段与推理批次均有终端进度提示。

`requirements_npu.txt` 是昇腾环境的依赖版本记录，包含平台相关包，应按目标服务器环境核对；CUDA 环境不应直接照搬其中的昇腾软件栈。昇腾 Transformer/TCN 适配与实测说明见[昇腾性能记录](docs/ASCEND_PERFORMANCE_20261015.md)。

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
- [训练参数与配置覆盖](docs/TRAIN_CONFIGURATION.md)
- [采样与训练/验证划分](docs/DATASET_SELECTION.md)
- [训练输出文件、指标与图片](docs/TRAINING_OUTPUTS.md)
- [数据增强方法、阈值与顺序](docs/DATA_AUGMENTATION.md)
- [数据集可视化工具](docs/DATA_VISUALIZATION.md)
- [各模型的训练与推理入口](docs/MODEL_ENTRYPOINTS.md)
- [独立测试集推理与多模型投票](docs/INFERENCE_AND_TEST_DATA.md)
- [断点续训](docs/RESUME_TRAINING.md)

代码测试可在已安装相应依赖的环境中运行：

```bash
python -m pytest -q tests
```
