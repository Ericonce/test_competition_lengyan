# 断点续训

独立入口 `train/<配置名>/train.py` 默认 `resume=False`，正常从头训练。

需要续训时，在 `main()` 的 `model.train(...)` 中修改已列出的参数（优先于 YAML）：

```python
return model.train(
    pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
    data=PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
    output_dir=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap",
    resume=True,
    resume_checkpoint=PROJECT_ROOT / "results/modular_multigrain_iq_synthetic/last_model.pt",
    epochs=5,  # 本次额外训练 5 轮
)
```

例如权重保存于第 20 轮，本次执行第 21–25 轮。若启用早停，可能提前结束；续训时早停等待计数重新开始。需要跑满可将 `early_stopping_patience` 设为大于本次 epochs。

## 保存与恢复的内容

- `last_model.pt`：每个完整 epoch 保存，推荐用于续训。
- `best_model.pt`：验证 Macro-F1 最佳轮次，默认用于推理，也可从该轮次恢复训练。
- 新断点保存模型、AdamW 状态、学习率调度器、AMP scaler、归一化统计、epoch、历史记录、随机状态以及对应的历史最佳断点。
- 从 last 恢复不会错把最后轮次当成最佳模型，指定新的输出目录也能保留此前最佳模型。
- checkpoint 用临时文件写完再替换，避免保存过程中断产生半个正常命名文件。

恢复优化器后会沿用已保存的学习率，不使用当前配置的 lr 重置它。数据类别数、特征顺序、长度和模型结构必须匹配；数据根目录可以迁移。归一化使用断点内保存的 mean/std。

历史权重若没有完整训练状态，仍可以恢复模型、归一化和轮次，其他状态重新初始化，并明确提示；不能把这类情况当作精确状态恢复。旧 CSV 历史记录若存在，会读取至断点所在 epoch。

断点是 epoch 级，不是 batch 级；训练在一轮中途被中断时，从最近完成轮次继续。跨 CUDA/NPU 或不同版本环境不承诺逐位一致的随机结果。此次真实续训已验证 CPU 与本机 CUDA，未重新连接服务器验证 NPU。

## 程序接口与命令行使用

```python
model.train(resume=True, resume_checkpoint="results/.../last_model.pt", epochs=5)
```

```powershell
python -m pulsecls train --pipeline configs/pipeline/modular_multigrain_iq_synthetic.yaml --resume --resume-checkpoint results/modular_multigrain_iq_synthetic/last_model.pt --epochs 5
```

包 CLI `python -m pulsecls train` 也支持同名参数。
