# 项目结构

## 1. 总体数据流

当前输入为载频 CF、脉宽 PW 和脉冲重复周期 PRI。每个 CSV 是一条脉冲序列，
长度可以不同。训练先按文件名选择序列，再加载被选中的 CSV；加载时按配置
补零、循环重复或裁剪到固定长度 L，并转换为 `[3, L]`。

```text
类别目录中的 CSV: [原始脉冲数, 3]
  -> 总数据池索引、按序列采样、训练/验证划分
  -> 读取所选 CSV、调整长度、训练集统计量标准化
模型输入: [B, 3, L]
  -> 所选网络的编码器及可选信号假设分支
流级表示: [B, D]
  -> 分类层
分类分数: [B, K]
```

B 是批量大小，L 是 sequence_length，D 由模型结构决定，K 是训练类别数。
文件夹推理按批读取平铺 CSV，不采样训练池，使用权重保存的标准化统计。
接入新数据集见 [操作指南](NEW_DATASET_GUIDE.md)。

## 2. 目录职责

项目按 Ultralytics 风格组织：一个门面对象、任务中立引擎、按 task 挂载
trainer/predictor，网络实现放在 `nn/`。

```text
pulsecls/
  __init__.py        PulseCLS 门面
  cfg/               default.yaml、get_cfg、CLI
  engine/            Model、BaseTrainer、BasePredictor、VotePredictor
  models/classify/   task_map：ClassificationTrainer / Predictor
  nn/                backbone、head、hypothesis IQ、MambaSL、build_model
  data/              CSV 加载、采样/划分、文件索引、合成生成器、序列增强
  signals/           三通道 PDW 条件化假设 IQ；不是实测 IQ 重建
  losses/            CE、InfoNCE、多粒度目标
  utils/             device、metrics、seed、终端进度

configs/
  model/             模型 YAML
  pipeline/          模型训练/推理配置（覆盖 default.yaml）
  augment/           增强方法、顺序、开关与阈值
  generator/         合成辐射源生成设置
  inference/         多模型投票默认配置

datasets/            各训练总池的 data.yaml、类别 CSV；独立平铺测试目录
results/             各训练结果与推理输出
train/               通用及各模型独立训练入口，调用 model.train()
infer/               独立、通用及投票推理入口
tools/               合成训练总池与配套测试集生成入口
docs/                使用说明、方法说明与历史实验记录
```

用户侧统一 API：

```python
from pulsecls import PulseCLS

model = PulseCLS()
model.train(
    pipeline="configs/pipeline/modular_multigrain_iq_synthetic.yaml",
    data="datasets/synthetic_radar_overlap/data.yaml",
    output_dir="results/modular_multigrain_iq_synthetic",
    num_classes=100,
    epochs=20,
    device="auto",
)
model.infer(
    pipeline="configs/pipeline/modular_multigrain_iq_synthetic.yaml",
    source="datasets/synthetic_radar_test",
    checkpoint="results/modular_multigrain_iq_synthetic/best_model.pt",
    save_dir="results/modular_multigrain_iq_synthetic/infer_test",
    labels_file="",  # 填写 labels.txt 路径才评估。
    num_classes=100,
)
```

或：

```powershell
python train/modular_multigrain_iq_synthetic/train.py
python infer/modular_multigrain_iq_synthetic/infer.py
```

旧导入（`pulsecls.engine.trainer.Trainer`、`pulsecls.models.build.build_model`、
`pulsecls.utils.config.load_pipeline`）仍可用，指向新位置。

## 3. 配置覆盖规则

`pulsecls/cfg/default.yaml` 提供全局默认。一个 pipeline YAML 再引用
数据集内的 `data.yaml` 和 `model` YAML，并可用：

```yaml
model_overrides: {}
train_overrides: {}
augment_overrides: {}
loss_overrides: {}
inference_overrides: {}
```

`get_cfg(..., overrides={"epochs": 3, "device": "auto"})` 会再覆盖 train 段。
最终配置写入实验目录的 `resolved_pipeline.json`。

train/infer 入口把路径与常用参数直接写在方法调用中，代码显式参数优先于 YAML。
详细增强方法和阈值在调用的 augment.preset 指向的 YAML 中设置。设备 auto
优先选择 CUDA，其次昇腾 NPU，最后 CPU。更多规则见 [参数设置](TRAIN_CONFIGURATION.md)。

## 4. 新增模块的方法

1. 网络层放到 `pulsecls/nn/`，并在 `nn/tasks.py` 注册。
2. 新 task 在 `pulsecls/models/<task>/` 提供 trainer/predictor，挂到 `task_map`。
3. 不要复制 `engine/` 里的训练循环，只覆盖钩子。
4. 在 `configs/model` 或 `configs/pipeline` 中提供可复现实验配置。
5. 为形状、配置解析和关键行为添加测试。
