# 昇腾训练与推理性能实测（2026-10-15）

测试在服务器 `/data/lym/test_competition_lengyan` 完成，使用 `/data/lym/radar_env`、Ascend910B3、PyTorch 2.1.0、torch_npu 2.1.0.post10。修复已部署服务器，并保存到本机项目。11 个核心代码文件的 SHA256 一致；服务器原有的数据采样比例等自定义参数保留。

## 结论与修改

- Transformer 原实现训练正常，验证时调用 `aten::_transformer_encoder_layer_fwd`，日志明确提示回退 CPU，每批需要约 25 秒。修复后训练、验证均使用 NPU 注意力，验证每批约 4.3 毫秒。这个巨大变化发生在验证阶段，训练本身的速度基本持平。
- NPU 编码器继续使用原模型的投影、注意力、残差、前馈与归一化权重；权重键名及形状没有改变。FP16/BF16 使用 `torch_npu.npu_fusion_attention`，FP32 使用设备上的矩阵运算。半精度矩阵注意力使用 FP32 中间累积。CUDA/CPU 沿用 PyTorch 编码器。
- TCN 的原生卷积在本机服务器没有发现 CPU 回退。测过拆分膨胀卷积和矩阵乘法两种等价实现，均更慢，因此默认保留原生卷积。
- 训练损失和预测先在设备汇总，默认每 10 批同步一次进度指标，结束时汇总准确率和 F1。每批完成数照常更新，耗时每秒刷新；终端阶段刷新限制为最多每秒 10 次。
- 训练提供 `amp_dtype` 参数。Transformer、TCN 的训练与推理配置使用 FP16；BF16 训练在本次测试中没有更快。推理分类概率转为 FP32 计算。

## 单批性能对照

输入为 batch=64、3 通道、长度 256、100 类，使用 AdamW 和 AMP。下表为预热后中位数，不含 CSV 读取和模型加载。原实现预热 3 批、计划测量 8 批；最终实现预热 6 批、测量 20 批。

| 模型及路径 | 训练每批 | 验证每批 |
|---|---:|---:|
| Transformer 原实现，FP16 | 26.00 ms | 25.68 s* |
| Transformer NPU 融合注意力，FP16 | 25.59 ms | 4.32 ms |
| TCN 原生卷积，FP16，修改前 | 17.28 ms | 3.28 ms |
| TCN 原生卷积，FP16，最终复测 | 17.51 ms | 3.16 ms |

* 原 Transformer 验证太慢，240 秒的有界测试超时结束。这里根据日志中预热后已完成的 5 批计算中位数；原始 JSON 只保存了训练结果，不能将其当作完成了全部验证测量。

其他候选的训练中位数：Transformer BF16 31.16 ms；TCN BF16 18.32 ms、phase 23.13 ms、linear 22.57 ms。未把这些较慢方案设为默认。短测存在运行波动，不应把 TCN 前后的小幅差异解释为稳定的模型算子加速。

## 现有数据完整链路检查

使用服务器已有合成数据，采样比例 1.0，6,400 条训练、1,600 条验证，长度 256，batch=64，训练 2 轮。数据缓存开启、增强关闭、绘图关闭；输出位于独立的 `results/npu_performance_20261015`，未覆盖正式模型结果。

| 模型 | 第 1 轮训练+验证 | 第 2 轮训练+验证 | 验证集推理完整调用 |
|---|---:|---:|---:|
| Transformer | 4.46 s | 3.40 s | 1.12 s / 1,600 条 |
| TCN | 10.46 s | 2.32 s | 0.99 s / 1,600 条 |

每轮计时包含训练、验证与指标计算，不含其后的权重落盘和训练结束分析；推理完整调用包含权重加载、缓存读取、预测及结果保存。首轮包含当前进程首次执行相关算子的开销。两轮训练完整调用总时间分别为 Transformer 10.52 s、TCN 14.76 s。

两种模型均完成验证集预测与评估，并在平铺的 100 条测试 CSV 上完成无标签推理，生成 `predictions.csv`。损失有限，训练正常下降。本次仅验证运行与性能，未进行长期收敛对比。

## 数值与设备核对

- 显式编码器和两种可选卷积实现的 CPU 前向、输入梯度、参数梯度测试通过。
- NPU 融合注意力与 NPU 矩阵注意力比较，前向最大绝对差约 `0.000977`，参数梯度最大绝对差约 `0.000215`。相对 FP32 参考，各参数的融合注意力梯度相对 L2 误差最大约 1.11%，梯度均有限。
- Transformer profiler 记录了 `FlashAttentionScore`、`FlashAttentionScoreGrad` 设备内核；TCN 记录了卷积前向和反向设备内核。两者跟踪均未调用 `_transformer_encoder_layer_fwd`，最终测试日志未出现 CPU 回退警告。Python 调度、CSV 读取及 sklearn 指标计算仍使用 CPU。
- 本地相关回归检查通过：训练、设备、数值、数据增强相关测试 183 项；设备、文件夹推理、投票及入口配置相关测试 63 项（两组包含重叠项）。

## 参数与复现

训练入口 `train/transformer_synthetic/train.py`、`train/tcn_synthetic/train.py` 提供：

```python
use_amp=True,
amp_dtype="float16",
progress_metric_interval=10,
```

对应推理入口和 pipeline 使用：

```python
inference={"use_amp": True, "amp_dtype": "float16", "batch_size": 256, "progress": True},
```

修改配置和代码后需重新启动训练进程。首次编译仍可能比后续批次慢。需要按批刷新 loss/acc 时，将 `progress_metric_interval` 设为 1，会增加同步开销。

单批性能测试示例：

```bash
source /data/lym/radar_env/bin/activate
cd /data/lym/test_competition_lengyan
python tools/benchmark_npu_sequences.py --model transformer --backend auto --dtype fp16 --warmup 6 --steps 20 --profile --output results/npu_recheck/transformer.json
python tools/benchmark_npu_sequences.py --model tcn --backend native --dtype fp16 --warmup 6 --steps 20 --profile --output results/npu_recheck/tcn.json
```

`tools/validate_npu_training.py` 是本次服务器数据的完整链路检查脚本，按当前 100 类、8,000 条数据和已生成测试文件夹设置断言；不是通用训练入口。常规训练继续使用各模型的 train.py。

服务器结果目录 `/data/lym/test_competition_lengyan/results/npu_performance_20261015` 保存修改前备份、单批测试 JSON/日志、profiler 跟踪、两轮训练输出与推理 CSV。本机 `results/npu_performance_20261015` 保存测试记录副本和 `test_records.tar.gz`；大型原始 profiler 跟踪留在服务器。
