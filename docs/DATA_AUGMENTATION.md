# 可组合的三维 PDW 数据增强

本次按用户选择接入变换与样本融合方法，不包含需要额外训练的 GAN、VAE 等生成模型。

## 最简单的控制方式

编辑 `configs/augment/default.yaml`，各 synthetic 训练入口均显式引用该文件；入口的 augment.enabled 为 False 时关闭训练增强，改为 True 后执行 YAML 中开启的方法：

- 顶层 `enabled`：总开关。
- `methods.<名称>.enabled`：每个方法自己的开关。
- `methods.<名称>.prob`：触发概率，0–1；普通方法按样本判断，MixUp 按批次判断。
- `std`、`ratio`、`keep_ratio` 等：各方法的可调强度/阈值。
- `order`：明确指定顺序。关闭的方法跳过，开启但未列入 order 的方法会报错，避免静默遗漏。

各方法是否开启以所选 YAML 的 methods.<名称>.enabled 为准；训练调用中的 augment.enabled 为 False 时，总开关优先关闭全部增强。要使用增强，先将调用总开关改为 True，再开启 YAML 中需要的方法。
默认参数是工程起点，不表示已在当前数据上找到最优强度。幅值参数作用在训练标准化后的数据上，不是 MHz/us 的物理阈值。

也可以在 `train.py` 的 `model.train(...)` 中追加 `augment`，局部设置优先于共享 YAML。
例如在默认配置基础上关闭其他默认增强，先抖动，再时间扭曲：

```python
return model.train(
    pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
    data=PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
    output_dir=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap",
    augment={
        "enabled": True,
        "methods": {
            "jittering": {"enabled": True, "prob": 1.0, "std": 0.02},
            "time_warping": {"enabled": True, "prob": 1.0, "std": 0.1, "knots": 4},
            "scaling": {"enabled": False},
            "pulse_mask": {"enabled": False},
            "feature_mask": {"enabled": False},
        },
        "order": ["jittering", "time_warping"],
    },
)
```

未列出的设置仍继承 YAML。如果曾在 YAML 开启其他方法，需要同时关闭它们或在 order 中列出它们。

实现依次执行 `x1 = jittering(x)`，`x2 = time_warping(x1)`，不会把两种方法各自作用在原样本后拼接。
交换 order 中的方法会交换执行顺序。一次可串联多种方法。

## 方法及参数示例

触发概率由各方法的 prob 设置；关闭方法即便概率为 1 也不会执行。下面给出参数起点示例，实际开关和数值以选用的 YAML 与调用覆盖项为准。

| 名称 | 含义 | 示例强度/阈值 |
| --- | --- | --- |
| `jittering` | 抖动/高斯噪声 | `std=0.01` |
| `scaling` | 幅值缩放 | `std=0.02` |
| `rotation` | CF/PW 平面小角度旋转 | `max_angle_degrees=5.0` |
| `magnitude_warping` | 幅值曲线扭曲 | `std=0.05, knots=4` |
| `white_noise_windows` | 局部白噪声窗口 | `std=0.02, window_ratio=0.2` |
| `permutation` | CF/PW/PRI 分段重排 | `segments=4` |
| `slicing` | 随机切片并恢复长度 | `keep_ratio=0.9` |
| `window_slicing` | 窗口切片 | `keep_ratio=0.9` |
| `time_warping` | 单调时间扭曲 | `std=0.1, knots=4` |
| `window_warping` | 局部窗口伸缩 | `window_ratio=0.2, scales=[0.8, 1.2]` |
| `time_shift` | CF/PW/PRI 循环平移 | `ratio=0.1` |
| `time_flip` | CF/PW/PRI 脉冲顺序翻转 | `无额外强度参数` |
| `frequency_warping` | 脉冲序列频谱扭曲 | `max_ratio=0.05` |
| `vtlp` | VTLP 风格分段频谱映射 | `max_ratio=0.05, pivot_ratio=0.8` |
| `frequency_slicing` | 频谱带切除 | `band_ratio=0.1` |
| `random_noise_boxes` | STFT 局部噪声框 | `std=0.02, time_ratio=0.2, frequency_ratio=0.1, n_fft=32` |
| `stft_mask` | STFT 时频掩蔽 | `time_ratio=0.1, frequency_ratio=0.1, n_fft=32` |
| `aaft` | AAFT 幅值调整傅里叶替代数据 | `无额外强度参数` |
| `iaaft` | IAAFT 迭代替代数据 | `iterations=10` |
| `sfcc` | SFCC 风格同类频带组合 | `bands=3, blend=0.3` |
| `rim` | 遮挡并线性填补（RIM 风格） | `mask_ratio=0.1` |
| `interpolation` | 同类样本线性插值 | `blend=0.3` |
| `spawner` | 带随机锚点的同类 DTW 融合 | `std=0.01, window_ratio=0.1, max_points=64` |
| `random_guided_warp` | RGW 同类随机引导扭曲 | `window_ratio=0.1, max_points=64` |
| `random_guided_warp_shape` | RGWs 局部形状描述引导 | `window_ratio=0.1, max_points=64, shape_radius=2` |
| `discriminative_guided_warp` | DGW 判别引导扭曲 | `window_ratio=0.1, max_points=64, candidates=3` |
| `discriminative_guided_warp_shape` | DGWs 局部形状描述判别引导 | `window_ratio=0.1, max_points=64, candidates=3, shape_radius=2` |
| `dtw_mix` | 同类 DTW 对齐后插值 | `blend=0.3, window_ratio=0.1, max_points=64` |
| `dba` | 同类 DTW 重心平均 | `iterations=2, neighbors=3, window_ratio=0.1, max_points=64` |
| `emda` | EMDA 风格同类延时/均衡混合 | `blend=0.3, gain_std=0.05, max_delay_ratio=0.05` |
| `statistical_style` | 同类均值/方差风格迁移 | `blend=0.2` |
| `sample_combining` | 同类 DCT 频带组合 | `bands=3, blend=0.3` |
| `pulse_mask` | 脉冲值遮蔽（包括 PRI） | `ratio=0.01` |
| `feature_mask` | 值通道遮蔽（包括 PRI） | `ratio=0.01` |
| `mixup` | 跨类 MixUp（混合分类损失） | `alpha=0.2` |

## 标签、PRI 与训练方式

输入为 `[B, 3, L]`，当前合成通道为 CF/PW/PRI。PRI 是脉冲重复周期，允许非单调变化；数值、频域、遮蔽和序列变换均作用于这三个通道。已移除累计 TOA 的跨度缩放、原点恢复和反射处理。保留的时间扭曲是对序列索引重采样。默认不使用这些较强变形。

同类融合只使用当前训练 batch 的同类样本，不读取验证集，也不跨类混合。一个类别在 batch 中只有一个样本时，该样本跳过同类融合。

MixUp 会混合类别损失，只用于普通监督分类；对比学习分支会明确提示并跳过 MixUp，其他同类融合照常执行。MixUp 必须位于所有已开启的同类融合之后；其后可以继续使用不依赖配对标签的变换。

所有增强仅在训练阶段使用，验证与推理不执行随机增强。旧 flat 参数接口仍兼容历史代码。

## 原文与项目实现的边界

这是一套针对三维 PDW 的工程适配，不把所有方法都宣称为对应论文的完整复现：

- 频域的频率指沿脉冲序列变化的频谱，不是直接改变真实载频或声学音高。
- `vtlp`、`emda`、`sfcc` 是适用于序列的风格适配。原音频/语音任务特定实现不直接移植到 PDW。
- `rim` 用遮挡及线性填补实现局部缺失模拟，不包含原论文的递归学习模块。
- `statistical_style` 是同类统计风格迁移，不是训练后的 SFM 声学映射模型。
- `dtw_mix` 是批内同类 DTW 插值，不是完整的少数类过采样 DTW-SMOTE。`dba` 是当前批同类样本的 DTW 重心。
- DTW 默认最多 64 个对齐点，用于控制批内计算成本；设为序列长度可使用完整分辨率。Shape 版本通过局部窗口描述子计算对齐距离。
- 幅值和时间曲线采用可调控制点的线性插值，不宣称采用原作者的全部样条细节。
- TS-DENSE、ECGAug、Shapelets-SMOTE、ASP、MappingLabel、StyleTime 等仅在原文分类图中给出名称且任务特定或需要额外模型/资料，本次没有冒用这些名称实现简单替代。

实现参考：[原作者时序增强方法说明](https://github.com/uchidalab/time_series_augmentation/blob/master/docs/AugmentationMethods.md)、[引导扭曲论文](https://arxiv.org/abs/2004.08780)、[傅里叶替代数据论文](https://arxiv.org/abs/0812.2380)。

## CUDA/NPU 和验证

无需增加第三方依赖。常用变换直接使用当前设备；NPU 的 FFT/STFT/DCT 和 DTW 路径会将相关数据转换到 CPU 计算再返回原设备，避免依赖未验证的复数 NPU 算子。默认频域及 DTW 方法关闭，开启后可能增加训练耗时。

已验证各方法在 CPU 与本机 CUDA 上保持形状、有限数值、无输入原地修改、PRI 通道参与增强，并测试多方法串联、开关、阈值校验、同类配对与训练标签处理。服务器未更新，也未声称新增方法已经通过真实 NPU 测试。
