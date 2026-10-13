# 多粒度对比学习

## 1. 两个粒度

### 流级

主干输出先池化为流级表示：

```text
raw input       [B, 3, 256]
ResNet feature  [B, 512, 8]
pool + linear   [B, 256]
projection      [B, 128]
```

两次随机增强后的同一条脉冲流是正样本，不同流是负样本。当前默认使用
`hybrid` 流级损失：

```text
L_stream = r L_instance + (1-r) L_supervised
```

`L_instance` 是两视图配对 InfoNCE，`L_supervised` 把同类别流也作为正样本。

### 脉冲级

主干的 `[B, 512, 8]` 转成 `[B, 8, 512]`，每个 token 是一个脉冲位置的
512 维特征。条件分支从原始输入同步采样 8 个 `[CF, PW, PRI]`：

```text
raw pulse tokens       [B, 8, 512]
sampled PDW conditions [B*8, 3]
generated IQ           [B*8, 2, 64]
IQEncoder features     [B*8, 128]
```

因此脉冲级对齐的正样本是同一个采样位置的：

```text
raw pulse feature <-> its CF/PW/PRI-conditioned IQ feature
```

每次增强视图都独立生成一组配对特征，最终对两个方向的交叉熵取平均。
默认的 `pulse_condition` 是严格一一配对的跨模态 InfoNCE。

## 2. 总损失

当启用多粒度训练时：

```text
L_total =
  lambda_ce      L_ce
  + lambda_stream L_stream
  + lambda_pulse  L_pulse
  + lambda_adv    L_adv
```

当前 modular_multigrain_iq_synthetic 完整配置的 YAML 默认：

```text
lambda_ce = 1.00
lambda_stream = 0.10
lambda_pulse = 0.02
lambda_adv = 0.00
```

CE 直接优化分类 logits。InfoNCE 只优化投影头及其上游特征，不直接替代
分类损失。

上述数值对应该方法的 pipeline；其他方法以各自配置为准，调用中的 loss 覆盖项优先。新数据集保持三通道格式即可复用模型，接入步骤见 [新数据集指南](NEW_DATASET_GUIDE.md)。

## 3. 为什么需要两个粒度

- 流级对比学习约束整条脉冲流的类别结构和增强不变性；
- 脉冲级对齐约束单个 PDW 与其假设 IQ 视图的一致性；
- IQ 分类分支把 8 个假设 IQ 脉冲聚合成流级条件特征，并与原始流特征融合；
- 三者的作用可以通过因素消融分别测量，不能只看完整模型的总效果。

## 4. 训练和推理差异

训练时若启用对比学习，输入会生成两次增强视图：

```text
x -> augment_a -> model -> out_a
x -> augment_b -> model -> out_b
```

验证和推理不使用随机增强，条件采样改为固定的均匀位置，分类 IQ 分支也
默认使用确定性调制选择。这样验证结果不会因生成的随机假设而漂移。

## 5. 与真实 IQ 的关系

这个项目的 IQ 不是测量值，因此脉冲级损失的含义是“PDW 特征和条件化波形
假设的一致性学习”，而不是“真实 IQ 重建损失”。如果 IQ 假设与数据集类别
没有稳定关系，IQ 分支可能无效甚至伤害性能；这正是 `iq_v*` 机制消融和
`factor_*` 模块消融需要验证的内容。
