# 测试集生成、CSV 推理与模型投票

训练输入使用 `data.yaml`；测试推理直接指定 CSV 文件夹，无需测试用的 `data.yaml`。

接入其他来源的数据集时，先按 [新数据集接入指南](NEW_DATASET_GUIDE.md) 建立训练描述和类别编号。下方生成步骤用于项目合成数据；已有测试 CSV 可以直接进入单模型推理步骤，无需生成器元信息。

## 配套生成测试集

`configs/generator/synthetic_radar.yaml` 新增：

```yaml
test_num_sequences: 10000
test_output_dir: datasets/synthetic_radar_test
```

先生成训练/验证的总数据，再生成测试数据：

```powershell
python .\tools\generate_synthetic_radar.py
python .\tools\generate_synthetic_radar_test.py
```

第一条命令生成训练/验证总池，采样与划分比例仍在 pipeline 中设置。第二条命令只读取总池的 `emitters.json` 和生成配置，不读取训练 CSV，复用相同辐射源身份、参数、长度分布和噪声设置，用独立随机流生成测试序列。按类别尽量均匀分配，余数类别多一条，然后打乱顺序。测试数量小于类别数时仅覆盖部分类别。

如果生成 YAML 的种子、辐射源参数或噪声设置与训练快照不一致，测试生成器会指出不一致字段。可以用原训练池的 `generation_config.yaml` 作为 `--config`，也可重新生成训练池后再生成测试集。训练 CLI 使用了数量/噪声覆盖项时，应选择该训练快照。

目录和文件名称使用英文或数字，测试集没有类别子目录：

```text
datasets/synthetic_radar_test/
  0.csv
  1.csv
  ...
  labels.txt
  emitters.json
  generation_config.yaml
  sequences.jsonl
  summary.json
```

每个 CSV 对应一条随机长度序列，列名为 `CF_MHz,PW_us,PRI_us`。文件名不暴露类别。`labels.txt` 每行是 `文件名<TAB>类别编号`，例如 `0.csv<TAB>17`；推理按文件名匹配标签，而不是依赖 TXT 行顺序。生成测试集不建立 `data.yaml` 或 `clean/` 子目录。

再次生成会替换同一测试目录，生成失败时保留旧数据。训练池与测试集必须使用独立目录。可以覆盖测试数量和输出目录：

```powershell
python .\tools\generate_synthetic_radar_test.py --num-sequences 2000 --output-dir datasets/test_small
```

## 单模型推理

`infer/cnn_softmax_synthetic/infer.py` 将 pipeline、CSV 输入文件夹、checkpoint、save_dir 直接写在 `model.infer(...)` 中，并列出 num_classes、可选标签路径、设备、输入长度、补长策略、批量大小、进度开关和增强 YAML 引用。代码参数优先；删除对应参数可继承 YAML。增强引用只用于配置，推理不执行随机增强。

```python
model.infer(
    pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
    source=PROJECT_ROOT / "datasets/synthetic_radar_test",
    output_dir=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap",
    num_classes=100,
    augment={"preset": PROJECT_ROOT / "configs/augment/default.yaml"},
    checkpoint="results/cnn_softmax_synthetic_overlap/best_model.pt",
    labels_file="datasets/synthetic_radar_test/labels.txt",
    save_dir="results/cnn_test",
    inference={"batch_size": 256},
)
```

独立推理入口显式指定 `best_model.pt`；无验证模式训练后，请将 checkpoint 改为 `final_model.pt`。删除 checkpoint 参数后可按 pipeline 自动选择权重。推理结果目录使用 save_dir，修改它不会影响显式权重路径。

统一 CLI：

```powershell
# 只预测；即使目录已有 labels.txt，也不会自动读取标签。
python .\infer\infer.py --source datasets/synthetic_radar_test

# 预测并评估。
python .\infer\infer.py --source datasets/synthetic_radar_test --labels-file datasets/synthetic_radar_test/labels.txt
```

每次生成 `predictions.csv`：`filename,path,prediction,confidence`。指定标签后增加 `label,correct`，同时输出 `metrics.json`、混淆矩阵和分类报告，指标包含 `macro_f1`。不指定标签时不计算准确率或 F1，也会清理同一输出目录中此前该模式的评估文件，避免误认旧结果。

推理按批读取 CSV，不加载训练/验证集、不使用训练采样比例、不做随机增强。输入长度、补零/重复及裁剪策略由 pipeline 的 `sequence_length`、`length_policy` 提供默认值，入口显式参数优先；标准化使用检查点保存的训练均值与标准差。pipeline 引用的数据描述 YAML 与模型 YAML 仍须存在以解析配置。

文件夹推理的特征列顺序与模型结构以检查点保存的训练配置为准；类别数默认读取权重，手动设置 inference.num_classes 时优先采用该值。减少时保留前 N 个标签；增加时新增未训练类别概率为 0。训练时如果用 `data=小型数据集/data.yaml` 覆盖原 pipeline 的大数据集配置，推理仍可使用原 pipeline；指定小型数据集对应的权重、测试文件夹及可选标签即可，不必把原 pipeline 的类别数改成小数据集类别数。终端会打印实际检查点的类别数。划分评估仍要求特征与权重一致；自动类别数模式要求数据类别配置匹配权重。显式设置 inference.num_classes 时，划分评估同样采用配置的输出范围，真实标签须位于该范围内。

终端默认显示 `Read input CSV`（完成读取的序列数）和 `Inference`（完成预测的序列数）两条进度，包括百分比、速度、耗时和预计剩余时间；最后一个不满 batch 的批次也按实际序列数计数。加载权重、初始化模型、写预测 CSV 和计算指标时显示阶段名称与耗时，每秒刷新。权重/压缩缓存读写无法准确预知百分比，因此这些步骤显示 `working`，不伪造完成比例。

需要关闭推理进度时，在调用中追加 `inference={"progress": False}`；投票可设置 `vote.yaml` 的 `progress: false`。训练的数据索引、采样文件检查、CSV 加载、缓存读写与标准化同样显示进度，可使用 `data_cfg={"progress": False}` 关闭这些数据准备提示。显示模块使用项目已有的 `tqdm`，无需增加其他依赖。

## 多模型投票

投票入口也在 `model.infer_vote()` 中直接列出常用参数和每个模型的 pipeline/checkpoint/开关。调用参数优先于 `configs/inference/vote.yaml`；删除对应参数可继承 YAML，或使用 `--config` 显式选择配置。配置字段：

- `source`：CSV 文件夹。
- `labels_file`：可选标签 TXT；`null` 只推理。
- `main_model`：主模型名称，必须处于开启状态。
- `models`：为每个成员设置唯一 `name`、`enabled`、`pipeline` 和 `checkpoint`。
- `device`、`sequence_length`、`length_policy`：可统一覆盖各成员的设置。

```powershell
python .\infer\infer_vote.py
python .\infer\infer_vote.py --source datasets/synthetic_radar_test --labels-file datasets/synthetic_radar_test/labels.txt
```

各成员必须在同一套辐射源及标签编号上训练；现有 TSRD/真实数据权重不能直接加入合成数据投票。示例中只有 CNN 默认开启，其他成员是关闭的配置示例，需要先在相同合成数据上训练并填写对应权重。投票成员全部引用对应 synthetic pipeline，测试输入始终来自 `source` 文件夹。

选择票数最多的类别；最高票并列时，主模型预测若属于并列候选则选择主模型，否则选择并列候选中编号最小的类别。所有模型预测均不相同时，必然采用主模型预测。

输出同样为 `predictions.csv`，增加各成员预测、`vote_count`、`vote_fraction`、`tied`、`main_prediction` 和 `main_confidence`。投票文件的 `confidence` 等于 `vote_fraction`，表示一致程度，不是模型概率；`main_confidence` 是主模型自身预测的概率。指定标签后评估最终投票结果。

## PRI 与旧 TOA 配置

当前合成输入第三通道是 PRI。已移除 `toa_relative` 配置；它原来表示从 TOA 中减去首个到达时间，PRI 不执行这一操作。

增强不再提供 TOA 专用 `time_scale`，也不保留/反射第三通道的累计到达时间坐标。保留的 `time_warping`、`window_warping` 等是序列索引重采样，作用于 CF/PW/PRI；PRI 可以参与噪声、缩放、遮蔽和样本融合，不要求递增。仅历史真实数据中显式请求 TOA 列的读取仍保留兼容处理；不会把该列误当 PRI。
