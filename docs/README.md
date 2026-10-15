# 项目使用文档目录

按用途选择下面的中文标题即可打开对应说明。模型名称、代码参数和文件路径保留原名，方便对照代码。

## 项目与模型

| 文档 | 内容 |
| --- | --- |
| [项目结构](PROJECT_STRUCTURE.md) | 总体数据流、目录职责、配置覆盖规则和新增模块的方法 |
| [模型训练与推理入口](MODEL_ENTRYPOINTS.md) | 各模型对应的流程配置、训练脚本和推理脚本 |
| [经典序列分类模型](CLASSIC_SEQUENCE_MODELS.md) | ResNet、TCN、LSTM、Transformer 等模型的结构与配置 |
| [一维残差网络分类模型](RESNET_SOFTMAX.md) | ResNet1D-18 + Softmax 的网络结构和使用方法 |
| [多粒度对比学习](MULTIGRAIN_CONTRASTIVE_LEARNING.md) | 序列级、脉冲级对比学习与损失组合 |
| [条件化假设 IQ 信号生成](IQ_SIGNAL_GENERATION.md) | 假设信号公式、调制方式与代码位置 |

## 数据准备与分析

| 文档 | 内容 |
| --- | --- |
| [新数据集接入指南](NEW_DATASET_GUIDE.md) | 数据目录、data.yaml、流程配置以及训练与推理的接入步骤 |
| [三通道输入说明](THREE_FEATURE_INPUT.md) | 载频、脉宽、脉冲重复周期及输入长度处理 |
| [合成辐射源数据生成引擎](SYNTHETIC_RADAR.md) | 生成参数、输出格式、噪声与参数重叠 |
| [总数据采样与训练验证划分](DATASET_SELECTION.md) | 先采样再划分，以及全量训练的设置 |
| [数据增强说明](DATA_AUGMENTATION.md) | 增强开关、参数、执行顺序与适用范围 |
| [数据集可视化与分析](DATA_VISUALIZATION.md) | 指定 data.yaml 查看序列与数据分布 |
| [合成数据难度审计记录](SYNTHETIC_DATASET_AUDIT.md) | 数据难度调整、实验协议与重训记录 |

## 训练、推理与结果

| 文档 | 内容 |
| --- | --- |
| [训练与推理参数设置](TRAIN_CONFIGURATION.md) | 常用参数、配置优先级、类别数、混合精度与进度显示 |
| [断点续训](RESUME_TRAINING.md) | 检查点路径、额外训练轮数与恢复内容 |
| [训练结果输出](TRAINING_OUTPUTS.md) | 指标、曲线、混淆矩阵、特征可视化与配置快照 |
| [测试集生成、推理与模型投票](INFERENCE_AND_TEST_DATA.md) | 平铺 CSV 测试集、可选标签、预测结果和投票 |
| [昇腾训练与推理性能实测](ASCEND_PERFORMANCE_20261015.md) | Transformer 与 TCN 的修复、耗时对照和验证记录 |
