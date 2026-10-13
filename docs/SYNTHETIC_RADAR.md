# 合成辐射源数据生成引擎

生成新的合成数据后如何配置训练/推理，以及其他来源数据的接入方法，见 [新数据集接入指南](NEW_DATASET_GUIDE.md)。生成器会写出数据描述，训练入口显式填写的类别数和路径仍需同步修改。

入口为 `tools/generate_synthetic_radar.py`，配置为 `configs/generator/synthetic_radar.yaml`，实现为 `pulsecls/data/synthetic.py`。生成器在 CPU 上运行，训练和推理仍使用项目现有的 CUDA/NPU/CPU 自动选择。

入口文件开头显式提供 `GENERATOR_YAML = PROJECT_ROOT / "configs/generator/synthetic_radar.yaml"`。使用自己修改后的配置文件时，直接把这一行改成相应 YAML 路径；测试集生成脚本也提供同名变量，应选择相同配置。启动时终端会打印实际读取的 YAML 路径，命令行 `--config` 可覆盖该变量。

## 快速使用

```bash
python tools/generate_synthetic_radar.py
```

辐射源数量和每类序列数量按生成 YAML 设置，每条原始序列默认 128–512 个脉冲（漏脉冲后 CSV 行数会减少），每条序列按 noise.snr_db_range 独立抽取特征 SNR。修改 YAML 即可改变全部参数，也可以用命令行覆盖常用参数：

```bash
python tools/generate_synthetic_radar.py --output-dir datasets/synthetic_custom --num-emitters 50 --sequences-per-emitter 200 --length-min 100 --length-max 600 --snr-min 25 --snr-max 40
python tools/generate_synthetic_radar.py --output-dir datasets/synthetic_fixed --length 256 --fixed-length
```

同一路径再次运行会完整替换旧的合成数据，包括旧 CSV、划分索引和缓存。新数据先在临时目录生成，成功后再替换；生成失败保留原数据。已有非空目录须包含生成器的 generation_config.yaml 与 emitters.json，避免误替换项目或其他目录。同一配置和 seed 的 CSV 可复现。若改变辐射源数量，也要同步修改逐类数量列表（如果使用列表）；统一数量不需要修改列表。

## 类别与调制

每个类别对应一个具体辐射源。先随机选取发射机模板，再为辐射源固定基础 CF/PW/PRI、变化幅度、周期、码表。生成同类序列时随机长度、起始位置、小幅参数漂移和噪声，保持身份参数不变。

PRI 模式为 fixed（固定）、staggered（参差）、jittered（均匀抖动）、sinusoidal（正弦）、sliding（周期滑变）。

CF 模式按**脉冲序号**表达：linear_fm 为周期线性扫频，nonlinear_fm 为周期三次曲线扫频，frequency_coded 为循环离散频率码，phase_coded 为将二相码映射到载频高低档的代理。这里不生成脉内波形，也不声称三个描述字可以恢复真实相位码；其表示方式记录在 emitters.json 中。

PW 模式为 fixed、multilevel、sinusoidal。手动设置 transmitter.templates 可以指定模板组合，此时实际模板数量由列表决定，num_transmitter_types 只在随机模板模式使用。

```yaml
transmitter:
  templates:
    - {pri_mode: fixed, cf_mode: linear_fm, pw_mode: fixed}
    - {pri_mode: staggered, cf_mode: frequency_coded, pw_mode: multilevel}
```

## 输出格式

```text
datasets/synthetic_radar_overlap/
  0/0.csv
  0/1.csv
  1/0.csv
  ...
  generation_config.yaml   # 完整生成配置
  emitters.json            # 模板、每类固定参数与码表
  sequences.jsonl          # 每条序列长度、起点、目标/实际 SNR、能量
  summary.json             # 总数量、长度范围和噪声范围
  dataset_index.json       # 各类别的连续序列编号数量，支持直接采样编号
  data.yaml                # 总数据位置、通道和类别数，all: .
  # 不再生成 train/val/test 划分索引；pipeline 决定运行时的采样和训练/验证划分
```

CSV 的列名为 CF_MHz、PW_us、PRI_us，每行对应一个脉冲。PRI 是该脉冲到下一脉冲的间隔，不是累计到达时间。所有生成的物理值为正，PW 小于 PRI × max_duty_ratio。

生成器只输出完整的类别目录和总数据元信息，不划分测试集。采样比例及训练/验证比例在 pipeline 的 dataset 中配置，先采样后划分，具体说明见 [采样与划分](DATASET_SELECTION.md)。

## 能量与噪声

完整干净序列 X 先按固定 channel_scales 得到 Z，计算 E_s=sum(Z²)。目标噪声能量为 E_n=E_s/10^(SNR/10)。生成标准高斯随机数，再按目标能量整体缩放，最后恢复各通道单位。固定尺度避免直接把 MHz 和微秒的平方相加；同一数据集使用相同尺度。

SNR 是三通道描述字的数值能量比，不是接收波形 SNR。默认 power_allocation=per_channel，将完整序列的目标噪声能量按各通道自身信号能量分配，使每个通道在物理修正之前都达到相同目标 SNR，避免小脉宽被其他通道分配来的噪声淹没。equal 保留旧的等方差噪声方式，各通道 SNR 可不同。噪声来自高斯随机抽样，有限序列经能量缩放，包含基础参数的完整序列能量参与计算，固定 PRI 也会得到噪声。

默认 invalid_policy=clip：加噪后将非正值修正为正数，将超过 PRI × max_duty_ratio 的 PW 限制在边界之内，支持 10–20 dB 等低 SNR 配置。修正后噪声不再是未经约束的高斯噪声，实际 SNR 可能偏离目标，绝不把目标值当成修正后的实测值。sequences.jsonl 保存目标 snr_db、实际 achieved_snr_db、修正数量 corrected_values 和实际噪声能量。

invalid_policy=resample 保留严格重试模式，不修正数值，要求整条序列满足物理约束；低 SNR 仍可能在 max_attempts 后报错。error 表示第一次出现非物理值就报错。save_clean=true 可保存干净 npy，独立核对能量比。

## 训练与推理的固定长度

在 `configs/pipeline/cnn_softmax_synthetic.yaml` 中设置（训练和推理共同使用）：

```yaml
sequence_length: 256
length_policy:
  short: zero  # zero=尾部补零；repeat=循环重复整条序列；error=报错
  long: head   # head=保留前段；center=中心裁剪；error=报错
```

补零行不参与训练归一化统计，标准化后仍保持全零；repeat 会将原序列按原顺序重复并截取到目标长度。缓存区分长度及补齐/裁剪策略，切换策略不会混用缓存。模型接收固定长度的三通道张量，未增加模型内部的 padding mask。

生成配置只负责原始序列的长度分布；生成器不写入训练/推理输入长度，也不补齐或裁剪原始 CSV。修改 pipeline 中的输入长度与策略后，直接重新运行训练或推理即可，无需重新生成数据。代码中的 sequence_length/length_policy 或 data_cfg 覆盖项仍优先于 pipeline。

训练：`python train/cnn_softmax_synthetic/train.py`。推理：`python infer/cnn_softmax_synthetic/infer.py`。

也可在调用中覆盖设置，优先于 YAML：

```python
from pulsecls import PulseCLS

pipeline = "configs/pipeline/cnn_softmax_synthetic.yaml"
data = "datasets/synthetic_radar_overlap/data.yaml"
model = PulseCLS()
model.train(pipeline=pipeline, data=data,
            data_cfg={"sequence_length": 256, "length_policy": {"short": "repeat", "long": "head"}})
model.infer(pipeline=pipeline, source="datasets/synthetic_radar_test",
              checkpoint="results/cnn_softmax_synthetic_overlap/best_model.pt",
              data_cfg={"sequence_length": 128, "length_policy": {"short": "zero", "long": "head"}})
```

CNN 示例支持上述长度切换；其他网络如含固定尺寸层，需要与训练时的尺寸要求一致。换成其他现有模型时，在相应 pipeline 中加入 sequence_length 和 length_policy，把 data 指向生成的 data.yaml；增强默认直接处理 CF/PW/PRI。使用补零时建议先关闭随机增强，再按需开启，时间变换可能改变零尾部的位置。

自定义训练池输出目录时，修改训练入口 model.train() 的 data 路径；生成的 data.yaml.root 相对于 YAML 所在目录，整体迁移数据文件夹仍可读取。

## Python 调用

```python
from pulsecls.data.synthetic import SyntheticRadarGenerator

SyntheticRadarGenerator(num_emitters=10, sequences_per_emitter=50,
                        sequence={"length_range": [100, 400]},
                        noise={"snr_db_range": [30, 40]}).generate("datasets/my_synthetic")
```

代码覆盖项优先于生成 YAML；未列出的配置继承默认值。

## 参数重叠与观测扰动

默认输出为 datasets/synthetic_radar_overlap，旧 datasets/synthetic_radar 保留作为宽参数范围对照。overlap.enabled=true 时，每 5 个辐射源共享相近的 CF/PW/PRI 中心、同一模板和接近的调制幅度；各源保留不同的周期、码表与参差档位。base_jitter_ratio 按 CF/PW/PRI 指定组内中心最大偏差，emitters_per_group 控制每组数量。

observation.enabled=true 时，每条序列有独立的整段测量偏差和漏脉冲率。session_bias_std_ratio 为三个通道的偏差标准差，持续作用在整个观测段，不会被逐点平均消除。漏掉脉冲后从累计到达时间重新计算 PRI，跨越漏脉冲的间隔等于多个原始 PRI 之和；不只是删掉 CSV 行。首尾脉冲保留。

noise.correlation_range 控制相邻脉冲高斯噪声的 AR(1) 相关系数。相关处理后按目标 SNR 缩放能量，物理修正后的实际值仍记录在 achieved_snr_db。sequences.jsonl 增加 group_id、generated_length、missing_pulses、requested_missing_rate、session_bias_ratios、noise_correlation，便于追溯。save_clean 保存的信号是在测量偏差和漏脉冲之后、高斯噪声之前的描述字。

完全关闭这些新增扰动可设置 overlap.enabled=false、observation.enabled=false、noise.correlation_range=[0, 0]；原始宽参数版本的现成数据及 pipeline 保留在 datasets/synthetic_radar 和 configs/pipeline/cnn_softmax_synthetic_baseline.yaml。

审计：python scripts/audit_synthetic_dataset.py。固定采样及训练/验证划分上三种训练种子复验：python scripts/retrain_synthetic_overlap.py。主要结果写入 results/cnn_softmax_synthetic_overlap，另外两种训练种子在带 seed 后缀的目录。审计与对照说明见 docs/SYNTHETIC_DATASET_AUDIT.md。

当前流程仅训练/验证；此前审计报告中的测试集数字是历史记录。新流程的 dataset_selection.json 保存实际采样及划分文件，旧划分索引已归档到 results/synthetic_dataset_audit/legacy_partitions。

独立测试集、文件夹推理和多模型投票见 [使用说明](INFERENCE_AND_TEST_DATA.md)。训练/验证池生成后，运行 `python tools/generate_synthetic_radar_test.py` 生成平铺测试 CSV 与 labels.txt。
