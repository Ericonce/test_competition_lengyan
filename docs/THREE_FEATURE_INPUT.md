# 三通道输入：载频、脉宽与脉冲重复周期（CF/PW/PRI）

当前模型入口与 pipeline 默认使用合成数据，也支持按相同格式接入 [新数据集](NEW_DATASET_GUIDE.md)。三个输入通道按 data.yaml 的 features 指定顺序读取 CF_MHz、PW_us、PRI_us，网络输入为 [batch, 3, length]。按列名读取，不将标签当特征。

PRI 为脉冲重复周期，允许固定、参差、抖动、正弦和滑变，不需要减去首值或保持递增。生成器不生成 IQ；部分分类模型会构建假设 IQ 视图作为内部特征，这不改变原始三通道输入。

输入长度及补零/循环重复、裁剪策略由各 pipeline 的 sequence_length、length_policy 提供默认值，train/infer 调用中显式设置的参数优先。补零行不参与训练归一化统计，标准化后仍为零；推理使用训练检查点保存的均值与标准差。

数据增强的数值扰动、遮蔽与序列变换按 CF/PW/PRI 处理，详细开关见 [增强说明](DATA_AUGMENTATION.md)。

所有完整模型都有 synthetic 的 [训练与推理入口](MODEL_ENTRYPOINTS.md)。测试推理直接指定 CSV 文件夹，标签 TXT 可选，见 [测试与推理说明](INFERENCE_AND_TEST_DATA.md)。
