# 通用工具

本目录保存与模型和数据无关的通用辅助代码。

## Python 文件

| 文件 | 作用 |
| --- | --- |
| `pyutils.py` | 保存命令行配置、同步输出终端与日志文件、累计均值、计时、线程批处理，以及生成局部位置配对索引。 |

## 用法

训练脚本已经自动使用 `save_config()` 和 `Logger`：

```python
from tool import pyutils

pyutils.save_config("./result/run_0", args)
logger = pyutils.Logger("./result/run_0/train.log")
try:
    print("这行内容同时写到终端和日志")
finally:
    logger.close()
```

累计多个标量的平均值：

```python
meter = pyutils.AverageMeter("loss", "acc")
meter.add({"loss": 0.5, "acc": 0.8})
meter.add({"loss": 0.3, "acc": 0.9})
mean_loss, mean_acc = meter.get("loss", "acc")
```

`Timer` 可统计总耗时和阶段耗时；`BatchThreader` 用线程池预取一组独立任务。`get_indices_of_pairs()` 与 `get_indices_of_pairs_circle()` 用于生成二维邻域位置索引。
