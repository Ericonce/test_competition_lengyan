# 总数据采样与训练/验证划分

训练总池保存类别目录和 data.yaml（all: .），训练/验证从这个池中采样与划分；独立测试集另放目录。项目生成器会产生这种格式，其他来源的数据可按 [新数据集接入指南](NEW_DATASET_GUIDE.md) 整理。pipeline 中提供以下默认设置，入口显式传入的 dataset 参数优先：

```yaml
dataset:
  sample_ratio: 0.5
  train_ratio: 0.8
  val_ratio: 0.2
  seed: 2026
  stratified: true
```

先按完整 CSV 脉冲流采样总数据的 50%，然后将采样结果的 80% 用作训练、20% 用作验证。总数据 10000 条时得到采样 5000、训练 4000、验证 1000；未采到的文件不会加载成训练/验证张量。总数据 800000 条、sample_ratio=0.01 时，采样 8000、训练 6400、验证 1600。合成数据按生成元信息直接抽取整数序列编号，不枚举全部 CSV。普通数据首次只建立文件名索引，后续复用。只有被采样的 CSV 才会进行文件属性检查和内容加载。

sample_ratio 的范围是 (0,1]；train_ratio 的范围是 (0,1]，val_ratio 是 [0,1)，两者之和必须为 1。整数条数会取整，验证数为采样数减去训练数。比例是序列文件的数量比例，不是某一条脉冲流的脉冲数量比例。

seed 独立于模型训练种子；改变训练随机种子不会改变数据划分。仅改变 train_ratio/val_ratio 时，第一阶段采样池不变。stratified=true 按类别分层，以最大余数分配保持总量精确；在样本足够时保证类别在训练/验证中出现。采样过少时可能缺失类别；训练数量不足以覆盖采样类别或某个启用划分没有样本时会明确报错。可以增大采样比例或调整划分比例。

缓存区分采样设置、划分设置、固定输入长度、补齐/裁剪策略以及采样文件的大小/修改时间和目录索引版本。修改采样比例不会读到旧比例的缓存。最后一个训练 batch 仅有一条时会并入前一批，避免 BatchNorm 报错，保留全部采样数据。训练输出的 dataset_selection.json 记录总条数、采样条数、训练/验证条数及每个实际使用文件。归一化统计只在当前训练集合的有效脉冲上拟合。

## 调参后的全量训练

```yaml
dataset:
  sample_ratio: 1.0
  train_ratio: 1.0
  val_ratio: 0.0
  seed: 2026
  stratified: true
```

这样使用总数据的 100% 训练。如果 sample_ratio=0.5、train_ratio=1.0，则只用采样出的 50% 训练。

val_ratio=0 时不创建/读取验证加载器，不执行验证、基于验证的学习率调度或早停，按 epochs 跑完本次轮数。保存 last_model.pt（可续训）及 final_model.pt（最后一轮权重），不保存按验证选择的 best_model.pt。同一输出目录中残留的旧 best_model.pt 和验证分类报告会清除，避免混用。

日志和 final_metrics.json 明确标注 validation_enabled=false、evaluation_split=train；这些是训练数据指标，不是验证指标。建议最终全量训练使用新的 output_dir，便于保留调参阶段的结果。resume 默认关闭；需要继续已有权重可按项目的断点续训接口启用，epochs 仍表示额外轮数。

当前独立推理入口直接读取 source 指定的测试文件夹，checkpoint 已显式填写；无验证训练后须将其改为 final_model.pt。API 未指定 checkpoint 时才根据划分配置自动选 best_model.pt 或 final_model.pt。未指定 source 的划分推理默认使用验证集，无验证时使用训练集；此时显式请求 split=val 会报错。

## 代码覆盖

```python
model.train(
    pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml", data=PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml", output_dir=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap",
    dataset={"sample_ratio": 0.5, "train_ratio": 1.0, "val_ratio": 0.0},
)
```

代码设置优先于 pipeline，也支持 **{"dataset.sample_ratio": 0.5} 或 data_cfg={"selection": {...}} 的覆盖方式。

## 现有数据与历史实验

总池默认由 data.yaml.all 指定。现有真实数据没有 all 时，可以在 pipeline 加入 dataset，框架将现有 train/val/test 路径收集成总池、去重后重新采样和划分，不修改原始文件。

当前合成数据的旧划分索引已归档到 results/synthetic_dataset_audit/legacy_partitions，原始类别 CSV 不因协议变化而重新生成。此前含测试集的指标保留为历史记录；新划分要重新训练、记录验证结果，不与历史数字混用。生成大数据时仍直接运行 tools/generate_synthetic_radar.py。

## 加载顺序与大数据性能

训练启动时只生成一次采样/划分计划，训练加载器和验证加载器复用该计划，不重复扫描或重新采样。load_split_arrays 的 selection_plan 参数允许复用计划；缓存保存只包含当前划分的已选数据。每个 CSV 只打开一次，同时读取表头和数据。

旧 data.yaml 中残留 train/val/test JSON 路径时，只要数据根目录直接包含数字类别文件夹，就从完整类别池采样，忽略旧划分路径；无需恢复被删除的 splits/val.json。

生成器现在附带 dataset_index.json 和数量元信息。既有合成数据可从 generation_config.yaml 和 summary.json 读取数量，不需要重新生成 80 万条数据。普通数据的文件名索引保存于 .pulsecls_cache/file_index_*.json，类别目录或清单变化时更新；手动改变原始文件集合后可能需要一次索引刷新，但不读取未采样的 CSV 内容。
