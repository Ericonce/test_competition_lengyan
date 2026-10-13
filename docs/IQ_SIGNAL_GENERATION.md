# 条件化假设 IQ 信号生成

## 1. 目的和边界

项目没有真实 IQ 采样，也不会声称从 PDW 唯一恢复真实 IQ。当前模块的
作用是：把一个归一化的 `[CF, PW, PRI]` 条件映射为一组可微分的假设 IQ 视图，
让模型学习“PDW 条件和可能的波形结构之间”的一致性。

因此它是条件化信号假设和数据增强，不是物理测量数据的重建。

## 2. 基本公式

归一化时间轴为

```text
t_n = n / (L - 1),  n = 0, ..., L - 1
```

默认 IQ 长度 `L=64`。复包络写为

```text
s(t) = A(t) exp(j phi(t))
I(t) = A(t) cos(phi(t))
Q(t) = A(t) sin(phi(t))
```

当前软矩形包络为

```text
A(t) = sigmoid(k (w - t))
```

其中 `k=48`，`w` 是由 PW 条件得到的归一化脉宽。

## 3. CF/PW/PRI 到参数的映射

输入是训练集统计量归一化后的 `cf`、`pw` 和 `pri`，并裁剪到 `[-6, 6]`：

```text
N_c = 2 + 14 sigmoid(cf)
w   = 0.25 + 0.74 sigmoid(pw)
K   = 2 + 10 sigmoid(0.5 cf + 0.25 pw)
eta = 0.5 + 3 sigmoid(0.25 cf - 0.5 pw)
phi_0 = 2 pi sigmoid(pri)
```

其中：

- `N_c` 是基准载频相位旋转的归一化周期数；
- `w` 控制脉冲包络持续时间；
- `K` 控制 LFM 的二次相位项；
- `eta` 控制 NLFM 的非线性相位扰动。
- `phi_0` 是第三通道 PRI 映射出的有界初相位假设，不是物理相位测量值。

这些量是明确的实验参数，不是从外部数据估计的物理标定值。

## 4. 六种调制假设

### LFM

```text
phi_lfm(t) =
  2 pi (N_c t + 0.5 K t^2) + phi_0
```

### NLFM

```text
m(t) = eta sin(2 pi (1 + w) t)
phi_nlfm(t) = 2 pi (N_c t + m(t)) + phi_0
```

### 四状态 FSK

令 `q(t)=floor(8t)`，频偏为

```text
delta_f(t) = (mod(q(t), 4) - 1.5) * 0.35
phi_fsk(t) = 2 pi ((N_c + delta_f(t)) t) + phi_0
```

### BPSK

```text
q(t) = floor(8t)
b(t) = mod(q(t), 2) pi
phi_bpsk(t) = 2 pi N_c t + b(t) + phi_0
```

### LFM-FSK

```text
phi_lfm_fsk(t) =
  2 pi ((N_c + delta_f(t)) t + 0.5 K t^2) + phi_0
```

### LFM-BPSK

```text
phi_lfm_bpsk(t) = phi_lfm(t) + b(t)
```

最终用对应 `phi(t)` 代入 `I/Q` 公式。

## 5. 三种信号模式

配置项 `hypothesis_signal_mode` 控制生成结果：

- `full`: `I=A cos(phi)`, `Q=A sin(phi)`；
- `envelope`: `I=A`, `Q=0`，只保留包络；
- `phase`: `I=cos(phi)`, `Q=sin(phi)`，去掉包络。

`random` 调制类型在 `randomize=False` 时使用配置列表中的第一个调制，
在 `randomize=True` 时为每个条件随机选择一种调制。训练中的对比学习视图
默认允许轻微频率、脉宽和初相位扰动；分类 IQ 分支默认保持确定性，便于
消融实验隔离 IQ 分支本身的作用。

## 6. 代码对应关系

```text
pulsecls/signals/conditions.py  CF/PW/PRI -> N_c, w, K, eta, phi_0
pulsecls/signals/envelope.py    软矩形包络
pulsecls/signals/lfm.py         LFM 相位
pulsecls/signals/nlfm.py        NLFM 相位
pulsecls/signals/fsk.py         FSK 相位
pulsecls/signals/bpsk.py        BPSK 相位
pulsecls/signals/lfm_fsk.py     LFM-FSK 相位
pulsecls/signals/lfm_bpsk.py    LFM-BPSK 相位
pulsecls/signals/generator.py   组装 I/Q
```

条件化分支从每条输入流均匀采样最多 8 个 PDW：

```text
[B, 3, 256]
  -> [B*8, 3] conditions
  -> [B*8, 2, 64] generated IQ
  -> IQEncoder
  -> [B*8, 128] pulse-level IQ embeddings
  -> mean over 8 pulses
  -> [B, 256] stream-level IQ embedding
```

三通道输入属于当前数据协议；生成出的 IQ 自身仍为 I/Q 两通道。新数据集也只需提供 CF/PW/PRI CSV，不要求提供 IQ，详见 [接入指南](NEW_DATASET_GUIDE.md)。
