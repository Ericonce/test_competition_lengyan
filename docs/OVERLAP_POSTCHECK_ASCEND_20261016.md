# 后检验脚本的真实昇腾环境验证

验证日期：2026-10-16。

`tools/postcheck_overlap.py` 已在实际 Ascend 910B3 容器中完成验证：项目导入正常、NPU 运算正常、35 项相关测试通过，基于容器现有雷达 CSV 构造的 30 个完整命令行样例全部核验成功。脚本中的匹配、SHA 和文件整理在 CPU 上执行，本次没有更改生产代码或为匹配增加 NPU 运算。

## 环境与代码来源

| 项目 | 实际值 |
| --- | --- |
| 系统 | Linux aarch64，glibc 2.28 |
| 硬件 | 2 张 Ascend 910B3，`npu-smi` 健康状态 OK |
| CANN Toolkit | 8.5.1，`/usr/local/Ascend/cann-8.5.1` |
| Python | 3.10.10，`/usr/local/bin/python` |
| PyTorch / torch-npu | 2.1.0 / 2.1.0.post10 |
| NumPy / pandas | 1.26.4 / 2.3.3 |
| PyYAML / tqdm | 6.0.2 / 4.67.1 |
| scikit-learn | 1.7.2 |
| SQLite | Python 自带 3.32.3 |

验证代码来自本地提交 `c8b141b` 上的工作区快照，包含尚未提交的后检验工具、测试和说明。上传压缩包的 SHA-256 在本地和容器内一致：

```text
6c38338df2ff0b56528253da71695b98721e7f4879ea5fbd8e85c02e2e230ee9
```

后检验脚本的 SHA-256 在本地和容器内也一致：

```text
23d1d886b84f8252a51989e95b01a988f3d0bf3db71e9117065fc9160d5df894
```

验证目录为 `/data/mh/validation/postcheck_20261016_iZd1yI/`，代码快照位于其中的 `repo/`。测试未覆盖已有工程文件或模型权重。pytest 9.0.3 仅安装在该目录下继承现有系统包的 `test_env/` 中，容器全局 Python、Torch、NumPy 和 CANN 包保持原样。

## 运行时检查

加载现有 CANN 的 `set_env.sh` 后，真实系统 Python 成功导入 `torch`、`torch_npu`、`pulsecls` 及相关依赖。

NPU 探针执行：

```python
x = torch.tensor([1., 2., 3.], device="npu:0")
result = (x * x).sum().cpu().item()
```

实际返回 `14.0`，`torch.npu.is_available()` 为 True，可见设备数为 2，设备名为 `Ascend910B3`。环境详情记录在 `environment.json`，硬件状态记录在 `npu_smi.txt`。

## 自动测试

使用真实容器软件栈和隔离的 pytest 运行器，共 **35 项通过，耗时 8.67 秒**：

- 后检验新增 27 项：文件 SHA、不同格式的数值相同、双向连续包含、跨类别冲突、非连续片段、float64 精度、最短长度、实际训练清单、异常输入和输出保护。
- 仓库已有 8 项相关测试：原始 CSV 读取、有限数值、采样与索引复用、旧路径映射、缓存迁移及权重数据根目录映射。

完整结果在 `pytest.log`，返回码在 `pytest_exit.txt`，值为 0。空 CSV 反例触发现有读取器的一条预期警告，并正确报错。本次没有运行全仓库测试或重新训练模型。

## 现有雷达 CSV 的命令行验证

参考数据取自容器已有目录 `/data/mh/Dataset/radar_jc/all`，共 10 个类别、982,308 行原始脉冲。验证目录通过符号链接建立仓库要求的数字类别目录；读取特征为该数据实际具有的载频、脉宽和脉冲到达时间三列。仓库标准 CF/PW/PRI 三列另由前述自动测试覆盖。

每类构造三种测试文件：

| 样例 | 数量 | 期望规则 | 实际结果 |
| --- | ---: | --- | --- |
| 原始训练 CSV 的完整复制 | 10 | `sha256_equal` | 10 个通过 |
| 原始训练序列的中间连续片段 | 10 | `test_in_train` | 10 个通过 |
| 在原始训练序列两端增加脉冲 | 10 | `train_in_test` | 10 个通过 |

通过真实系统 Python 启动独立子进程，执行了完整 `tools/postcheck_overlap.py` 入口。其命令、数据路径和返回码保存在 `real_csv_validation.json`，命令行输出在 `real_csv_cli.log`。完整验证过程保存在 `verify_real_csv.py`。

验证结果：

- 测试文件 30 个，唯一类别匹配 30 个，冲突 0 个，未匹配 0 个。
- 30 个预期的文件对及匹配规则全部存在，输出记录的连续片段逐行数值一致。
- 30 个归类标签均与来源类别一致，分类复制后的 SHA-256 全部一致。
- 10 个原始参考 CSV 在验证前后的 SHA-256 全部一致。
- 结果目录可按仓库的 `classified/<类别编号>/<原文件名>.csv` 结构读取。

这些是用于验证功能正确性的已知重叠样例，不表示模型在独立测试集上的分类指标。

## 容器既有依赖告警

容器全局 `python -m pip check` 返回 1，原始输出已保存在 `pip_check.log`：

- `op-compile-tool` 将 `getopt`、`inspect`、`multiprocessing` 声明为缺少的包；这三个模块属于 Python 标准库。
- `te` 声明缺少 `ml-dtypes`、`tornado`。
- `superkernel` 要求 `sympy>=1.10.1`，现有版本为 1.4。
- `op_test_frame` 与 `te` 的平台标记未通过 pip 的平台检查。

以上来自容器已有 CANN 软件栈。后检验脚本没有新增运行依赖；本次项目导入、实际 NPU 探针、35 项测试及真实 CSV 命令行流程均正常完成。为保持现有软件栈，本次没有修改这些全局供应商包，也没有尝试将标准库模块作为第三方包安装。

## 本地日志与远端保留位置

原始证据包已下载到本地 `results/ascend_postcheck_validation_20261016/`，包含环境信息、测试日志、命令行日志、匹配统计、代码快照和复现实验脚本。该目录沿用仓库 `.gitignore`，不进入版本提交。

证据包名：`ascend_postcheck_validation_20261016.zip`；SHA-256：

```text
5553f4b95e422427d4955459f7c078fed82532b07ea5d0c4bb825b9b9fc26637
```

包内 `evidence_sha256.json` 记录每个证据文件的哈希。远端原始结果保留在 `/data/mh/validation/postcheck_20261016_iZd1yI/real_csv_smoke/output/`。重复验证请填写新的输出目录，避免与已生成结果冲突。
