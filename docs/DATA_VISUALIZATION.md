# 数据集可视化与分析（通过 data.yaml 指定数据）

入口是 [tools/visualize_dataset.py](../tools/visualize_dataset.py)。分析训练/验证的总数据池，不需要 pipeline 或模型权重，也不执行训练。

## 最简单的使用方式

从项目根目录执行：

```bash
python tools/visualize_dataset.py
```

脚本默认读取 datasets/synthetic_radar_overlap/data.yaml。换数据集时，在脚本末尾 main() 的 visualize_dataset() 调用中修改 data 路径即可。参数直接写在调用里：

```python
visualize_dataset(
    data=PROJECT_ROOT / "datasets/new_radar/data.yaml",
    output_dir=PROJECT_ROOT / "results/dataset_visualization/new_radar",
    class_ids=None,
    sequences_per_class=20,
    max_sequences=2000,
    plot_classes=8,
    examples_per_class=2,
    pulses_per_sequence=256,
    max_plot_points=100000,
    seed=2026,
    bins=50,
    dpi=300,
    formats=("png",),
    show=False,
    progress=True,
    save_sampled_pulses=True,
)
```

data.yaml 的 root、root_relative_to_yaml、all 和 features 与训练采用相同的解析规则。root_relative_to_yaml=true 时，root 相对于 data.yaml 所在目录解析，整体迁移项目仍可读取。支持数字类别目录以及现有加载器支持的样本清单；其他来源的数据不需要生成器元信息。数据格式见 [新数据集指南](NEW_DATASET_GUIDE.md)。

## 可调参数

| 参数 | 含义 |
| --- | --- |
| `data` | 要分析的数据集 data.yaml 路径。 |
| `output_dir` | 图片与统计文件的保存目录；None 时自动使用 results/dataset_visualization/<data.yaml 所在目录名>。 |
| `class_ids` | None 分析所有有样本的类别；填 [0, 1, 2] 只分析这些类别。 |
| `sequences_per_class` | 每个类别最多抽取的 CSV 数量。 |
| `max_sequences` | 实际加载的 CSV 总数上限；至少覆盖所有选定类别，否则明确报错。 |
| `plot_classes` | 类别对比和原始序列图最多展示多少类，按选定类别编号从小到大取前几类。总池计数及统计表不受这个显示上限影响。 |
| `examples_per_class` | 每个展示类别最多叠加多少条原始序列，来源于已经抽取的 CSV。 |
| `pulses_per_sequence` | 每条已抽序列用于脉冲分布图的随机脉冲数上限；不影响完整序列统计或原始曲线。 |
| `max_plot_points` | 脉冲分布图和 sampled_pulses.csv 的总点数上限。 |
| `seed` | 随机种子。数据文件集合与参数相同时，可复现采样；抽序列和抽脉冲使用独立随机流。 |
| `bins` | 直方图分箱数量。 |
| `dpi` | 图片清晰度，默认 300。 |
| `formats` | 可选 png、svg、pdf；汇总页面总会保存 PNG。 |
| `show` | False 只保存，适合服务器；True 在有图形界面的本机打开图窗。 |
| `progress` | 是否显示文件读取、保存统计和绘图进度。 |
| `save_sampled_pulses` | 是否保存实际绘图脉冲的数据，方便复核或重新绘图。 |

默认每类最多 20 条、总计最多 2000 条。类别内均匀随机抽序列，当总上限不足时在选定类别间分配额度，保证每类至少一条；数量少的类别最多取其全部现有样本。文件名先被选定，只有选中的 CSV 内容才会加载。普通数据首次建立文件名索引，生成器数据可使用已有数量元信息，避免先读取全部数据再采样。

## 输出内容

```text
results/dataset_visualization/<数据集目录名>/
  index.html
  dataset_overview.png
  feature_distributions.png
  class_feature_means.png
  feature_relationships.png
  sequences/
    class_000.png
    class_001.png
    ...
  class_counts.csv
  sequence_stats.csv
  sampled_pulses.csv
  analysis_summary.json
```

- `index.html`：本地汇总页面，可用浏览器打开，集中查看图片、范围说明和统计表链接。图片路径相对页面，整体复制输出目录后仍可查看。
- `dataset_overview.png`：上图为完整总数据池的各类别序列数，下图为抽样序列的原始长度分布。
- `feature_distributions.png`：抽样脉冲的 CF/PW/PRI 原始数值直方图。
- `class_feature_means.png`：各展示类别的序列均值箱线图与实际采样点，观察类别内部变化和类别间差异。
- `feature_relationships.png`：CF/PW、CF/PRI、PW/PRI 的序列均值散点图，观察不同类别的参数重叠。
- `sequences/class_*.png`：每类若干条序列的三通道曲线，观察固定、参差、周期变化、噪声等时序结构。
- `class_counts.csv`：总池中有样本的各类别数量、实际抽取数量和是否参与分析。
- `sequence_stats.csv`：每条已抽序列的类别、相对文件路径、原始长度，以及各通道的均值、标准差、最小值、最大值和非正数数量。标准差使用 ddof=0。
- `sampled_pulses.csv`：实际用于分布图的原始脉冲，sequence_id 对应 sequence_stats.csv，pulse_index 从 1 开始。
- `analysis_summary.json`：数据来源、类别范围、采样参数、实际数量、图片清单和索引指纹；也记录 data.yaml 声明却没有样本的类别编号。

图中的序列长度和统计都是原始值，不采用训练的 sequence_length、补零、重复、裁剪、标准化或增强。类别均值图的一个点代表一条序列；箱体为中位数及四分位范围，须为 1.5 IQR，并叠加实际均值点，不代表置信区间。

每类上限意味着序列样本通常近似按类别均衡，不等于完整数据池的类别比例。脉冲分布还受序列内抽点和全局点数上限影响，不能将图中的频数视为全量脉冲频数。序列均值重叠图用于探索，不替代训练/验证或分类性能评估。

## 命令行示例

```bash
# 分析小型数据集，自动按 data.yaml 所在目录创建结果目录。
python tools/visualize_dataset.py --data datasets/synthetic_radar_overlap_small/data.yaml

# 指定其他数据集和结果目录，仅分析 0、1、2 类。
python tools/visualize_dataset.py --data datasets/new_radar/data.yaml --output-dir results/dataset_visualization/new_radar --classes 0 1 2 --sequences-per-class 50

# 只分析少量序列，并在有图形界面的本机显示图窗。
python tools/visualize_dataset.py --classes 0 1 --sequences-per-class 5 --show
```

CLI 显式选项优先于 main() 中对应的默认值；更多绘图参数直接在调用中调整即可。脚本使用项目已有的 NumPy、Matplotlib、PyYAML 和 tqdm，不需要安装新的绘图库，也不需要 CUDA/NPU。
