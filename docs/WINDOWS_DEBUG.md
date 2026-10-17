# Windows 10 工作站调试

适用：Windows 10 x64，独立 Conda 环境 `radar_win10`。Python 3.10.22、PyTorch 2.1.0 CPU；NumPy、SciPy、Pandas、Matplotlib、scikit-learn 等核心依赖与服务器 `radar` 环境保持相同版本。

## 已有环境时使用

在仓库根目录打开 Anaconda Prompt 或已经初始化 Conda 的终端：

```powershell
conda activate radar_win10
python tools/debug_windows_cpu.py
```

这会在 `results/windows_cpu_debug/<运行时间>/` 生成 3 类、共 24 条训练池样本和 9 条测试样本，完成 2 轮 CPU 训练、额外 1 轮续训、测试集推理和绘图。每次运行保存到新目录。此入口用于检查环境和跟踪代码，不用于评价模型精度。

## VS Code 断点调试

1. 用 VS Code 打开仓库文件夹，安装 Python 和 Python Debugger 扩展。
2. 按 `Ctrl+Shift+P`，选择 `Python: Select Interpreter`，选择 `radar_win10`；使用 Conda-pack 时选择实际解压目录下的 `python.exe`。
3. 在 `tools/debug_windows_cpu.py` 的 `model.train(...)`，或 `pulsecls/engine/trainer.py` 中设置断点。
4. 在“运行和调试”中选择 `Windows CPU: small train, resume and inference`，按 F5。
5. 调试其他文件时选择 `Python: current file`。可用 F10 单步执行、F11 进入函数，并在变量面板查看张量和参数。

## 在比赛工作站重建

准备 Windows x64 版 Miniconda，以及本仓库。在仓库根目录执行：

```powershell
conda env create -f environment-win10.yml
conda activate radar_win10
python -m pip check
python tools/debug_windows_cpu.py
```

以上安装需要联网。比赛现场离线时，使用提前携带的 Windows Conda-pack，按压缩包的使用说明解压、激活，首次运行 `conda-unpack`，然后执行 `python tools/debug_windows_cpu.py`。安装清单本身不含软件包；Linux ARM64 环境不能直接用作 Windows 环境。

## 调试现有训练入口

在已有 `train/<方法>/train.py` 中按需设置：

```python
device='cpu',
use_amp=False,
num_workers=0,
epochs=1,
```

同时调小模型、输入长度和批大小。`sample_ratio=0.01` 只使用数据池的 1%，样本少时提高比例，保证训练和验证覆盖各类别；也可直接使用上面的独立小规模调试入口。

## 本地和服务器的分工

- 本地 CPU 环境用于数据读取、归一化、模型逻辑、损失、训练/续训/推理、绘图和自动化测试。
- 昇腾驱动、CANN、`torch_npu`、HCCL 双卡及 NPU 性能检查在 Linux 昇腾服务器执行。Windows CPU 环境不包含这些组件。
- 仓库 MambaSL 可使用现有的纯 PyTorch 实现；可选的 `mamba_ssm` 加速扩展不包含在此环境。
- 本环境安装 CPU 版 PyTorch，服务器的 `requirements_npu.txt` 用于昇腾环境。

项目测试中与 CUDA/NPU 专门相关的用例，需在对应硬件和软件环境运行。Windows 小规模验证通过也不等于所有服务器硬件功能均已验证。
