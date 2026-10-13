> 历史实验记录：本文所述旧数据集的 pipeline 已清理。当前入口见 [synthetic 模型入口](MODEL_ENTRYPOINTS.md)，下方旧命令与配置路径仅作历史记录。

# 消融实验说明

## 1. 两组消融

### IQ 机制消融

配置位于 `configs/pipeline/ablation/ablation_experiments/iq_v*.yaml`：

| 配置 | 含义 |
|---|---|
| `iq_v0_no_iq` | 不使用 IQ 假设 |
| `iq_v1_full_lfm` | 完整 LFM 假设 IQ |
| `iq_v2_envelope_only` | 只保留包络 |
| `iq_v3_phase_only` | 只保留相位 |
| `iq_v4_mismatched` | 将条件在 batch 间错配 |
| `iq_v5_randomized` | 随机调制和参数扰动 |

运行器：

对应旧运行脚本已删除，此处仅保留历史实验设计说明。

### 模块因素消融

配置位于 `configs/pipeline/ablation/ablation_experiments/factor_*.yaml`。因素为：

1. IQ 假设分类分支；
2. 流级 InfoNCE；
3. 脉冲级 PDW-IQ InfoNCE。

对应关系：

| 配置 | IQ 分类 | 流级 InfoNCE | 脉冲级 InfoNCE |
|---|---:|---:|---:|
| `factor_baseline_tsrd` | 0 | 0 | 0 |
| `factor_stream_infonce_tsrd` | 0 | 1 | 0 |
| `factor_pulse_infonce_tsrd` | 0 | 0 | 1 |
| `factor_iq_classification_tsrd` | 1 | 0 | 0 |
| `factor_iq_pulse_tsrd` | 1 | 0 | 1 |
| `factor_stream_pulse_tsrd` | 0 | 1 | 1 |
| `factor_full_multigrain_iq_tsrd` | 1 | 1 | 1 |

运行器：

对应旧运行脚本已删除，此处仅保留历史实验设计说明。

快速验证可以先运行：

对应旧运行脚本已删除，此处仅保留历史实验设计说明。

## 2. 记录的指标

每个种子会记录：

- `macro_f1`、`macro_precision`、`macro_recall`；
- `weighted_f1`、accuracy、balanced accuracy；
- 训练集样本数不超过 `rare_threshold` 的类别的
  `rare_precision`、`rare_recall`、`rare_f1`；
- 验证集上 `zero_recall_classes`；
- 模型参数量；
- 每个类别的 precision、recall、F1 和 support。

默认 `rare_threshold=10`。如果 TSRD 的类别计数定义需要改变，可以通过
`--rare-threshold` 调整。

## 3. 输出文件

```text
results/ablation/factor_tsrd/
  ablation_runs.csv
  ablation_summary.csv
  ablation_class_metrics.csv
  experiment_definition.json
  <variant>/seed_<seed>/
    resolved_pipeline.json
    training_history.csv
    best_model.pt
    final_metrics.json
    infer_val/
      predictions.csv
      val_classification_report.csv
      val_confusion_matrix.csv
```

## 4. 结果判定

主要按 `macro_f1_mean` 排序；同时检查：

- `rare_f1_mean` 是否提高；
- `rare_recall_mean` 是否提高；
- `zero_recall_mean` 是否下降；
- 多种子标准差是否过大；
- 完整模型是否稳定优于单模块模型。

如果完整模型只提升 weighted-F1 而 macro-F1 和 rare-F1 下降，不能认为它对
类别不平衡任务真正有效。

## 5. 类别感知流级 InfoNCE 实验

### 5.1 实验目的

原来的流级对比学习把同一条脉冲流的两次增强视图作为正样本。这种构造
主要约束“增强前后保持一致”，并没有直接利用辐射源类别信息。本实验将
流级样本对改为：

- anchor：当前 batch 中的一条脉冲流；
- 正样本：与 anchor 标签相同的其他脉冲流；
- 负样本：与 anchor 标签不同的其他脉冲流。

对于每个 anchor，当前实现从同一个 batch 中随机选取**最多 2 个同类样本**
和**最多 2 个异类样本**。如果当前 batch 中缺少同类或异类候选，该 anchor
会被跳过。因此这是“batch 内类别感知配对”，还不是严格保证每个 anchor
都有 2+2 个候选的 class-aware sampler。

对应代码为：

- `pulsecls/losses/infonce.py` 中的 `classwise_pair_infonce`；
- `pulsecls/losses/multigrain.py` 中的 `stream_infonce.mode: classwise_pair`；
- `configs/pipeline/ablation/stream_classwise_experiments/` 下的两个实验配置。

对 anchor \(i\)，记正样本集合为 \(P_i\)，负样本集合为 \(N_i\)，相似度为
\(s_{ij}=\operatorname{sim}(z_i,z_j)\)，温度为 \(\tau\)，则损失为：

\[
\mathcal{L}_i =
-\log
\frac{\sum_{j\in P_i}\exp(s_{ij}/\tau)}
{\sum_{j\in P_i\cup N_i}\exp(s_{ij}/\tau)}.
\]

batch 中没有有效正负候选的 anchor 不参与该 batch 的流级损失。

### 5.2 控制变量

实验使用 TSRD 随机窗口 256 数据集，训练集 17,972 条，验证集 4,496 条，
69 个类别；模型结构、数据划分、20 个 epoch、batch size=64 和
`seed=2026,2027,2028` 保持一致。主要评价指标为 Macro-F1，同时报告
少样本类别 F1 和验证集零召回类别数量。

### 5.3 三种子结果

| 配置 | 流级损失 | IQ 分类分支 | 脉冲级 PDW-IQ 对比 | Macro-F1 | Rare-F1 | 零召回类别 |
|---|---:|---:|---:|---:|---:|---:|
| 旧流级方案 | 实例增强式 | 否 | 否 | 0.4501 ± 0.0901 | 0.1587 | 26.67 |
| 新流级方案 | 类别感知，最多 2+2 | 否 | 否 | 0.4736 ± 0.0565 | 0.1488 | 25.67 |
| 完整对照 | 无 | 是 | 是 | **0.5617 ± 0.0257** | **0.3429** | **19.67** |
| 新完整方案 | 类别感知，最多 2+2 | 是 | 是 | 0.5508 ± 0.0225 | 0.3095 | 20.33 |

其中“2+2”表示每个 anchor 最多使用 2 个同类正样本和 2 个异类负样本。
汇总结果位于：

```text
results/ablation/stream_classwise_tsrd_20ep/ablation_summary.csv
results/ablation/factor_tsrd_20ep/ablation_summary.csv
```

### 5.4 结论

1. 新的类别感知流级配对比旧的实例增强式流级配对更符合分类任务目标：
   Macro-F1 从 `0.4501` 提升到 `0.4736`，标准差从 `0.0901` 降到
   `0.0565`。
2. 但在完整模型中，加入类别感知流级损失后 Macro-F1 从 `0.5617` 降到
   `0.5508`，下降约 1.09 个百分点；Rare-F1 从 `0.3429` 降到
   `0.3095`，零召回类别平均数也从 `19.67` 增加到 `20.33`。
3. 因此，在当前 TSRD、CF/PW 输入、batch size=64 和损失权重设置下，
   类别感知流级 InfoNCE 不能证明对最终辐射源识别有正贡献。当前最佳
   方案仍应保留 IQ 分类分支和脉冲级 PDW-IQ 对比，去掉流级对比。
4. 新配对方式可以作为后续研究接口保留，但若要继续尝试，应优先实现
   class-aware batch sampler，保证每个参与对比的 anchor 都能稳定获得
   2 个同类和 2 个异类候选，再重新调节流级损失权重。
