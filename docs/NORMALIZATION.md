# CF / PW / PRI 归一化方案

所有模型共用以下预处理。默认保持 `zscore`；三个通道各自拟合参数，统计来自本次采样、划分后的训练集有效脉冲。补零行不参与拟合，变换后仍保持为零。统计使用裁剪、补长后的数据；循环重复补长产生的脉冲会参与统计。

| `method` | 计算方式 | 说明 |
|---|---|---|
| `zscore` | `(x - mean) / std` | 保留现有行为，按训练均值和标准差标准化。 |
| `minmax` | `a + (b-a) * (x-min)/(max-min)` | 默认范围 `[0,1]`，也可配置 `[-1,1]`。 |
| `maxabs` | `x / max(abs(x))` | 按训练最大绝对值缩放，保留零点。 |
| `robust` | `(x-median)/(Q75-Q25)` | 使用训练中位数和四分位距，分位范围可调整。 |
| `log_zscore` | 对 `g(x)=sign(x)*ln(1+abs(x))` 做 Z-score | 对正值采用 `ln(1+x)`；增强产生负值时也有定义。均值、标准差在变换后的训练值上拟合。 |
| `none` | `x` | 保留原始数值，便于对照。 |

分母用 `eps`（默认 `1e-6`）作为下限，防止恒定通道除零。Min-Max 的恒定通道映射到范围下界；Z-score、Robust 的恒定通道映射到零。各方案的实际分类效果需要通过同一验证集比较。

## 在现有训练入口中选择

在任意 `train/.../train.py` 的 `model.train(...)` 调用中添加一项即可：

```python
normalization="minmax",
```

使用 `[-1,1]` 范围：

```python
normalization={"method": "minmax", "feature_range": [-1, 1], "clip": False},
```

调整 Robust 的分位数：

```python
normalization={"method": "robust", "quantile_range": [10, 90]},
```

也可在各 pipeline 的 `train` 部分填写：

```yaml
train:
  normalization:
    method: minmax
    eps: 0.000001
    feature_range: [0.0, 1.0]
    quantile_range: [25.0, 75.0]
    clip: false
```

全局默认值在 `pulsecls/cfg/default.yaml`。代码中显式传入的参数优先于 YAML，也支持 `train={"normalization": "robust"}` 和 `**{"train.normalization.method": "maxabs"}`。

命令行训练支持 `--normalization minmax`；范围、分位数等详细设置使用 YAML 或 Python 字典。

## 训练、验证、推理和续训

- 训练先在原始数值上执行数据增强，再应用训练集拟合的固定归一化参数。
- 验证集和测试集只使用已经拟合的参数，不参与统计。
- `minmax` 默认 `clip=False`：验证、测试或增强值超出训练范围时，结果可以超出 `[a,b]`。需要限制范围时显式设置 `clip=True`；这个开关仅用于 Min-Max。
- 训练保存 `normalization.json`、`normalization.npz`，并把完整方案和统计写进每个模型权重的 `normalization` 字段。`mean/std` 保留原始训练值的均值与标准差；完整变换以 `normalization` 中的 `offset/scale/config` 为准。
- 普通推理、文件夹推理和投票成员均自动读取各自权重中的方案，不需要在推理入口再次指定。
- 续训恢复原权重的方案及统计，并覆盖本次调用的归一化设置。更换方案进行对照时，设置 `resume=False`，从头训练并使用不同结果目录。
- 没有 `normalization` 字段的旧权重按原有 `mean/std` 还原 Z-score，保持旧推理和续训兼容。

原始 CSV 及数据缓存保持原始物理数值；切换归一化方案不需要重建原始数据缓存。
