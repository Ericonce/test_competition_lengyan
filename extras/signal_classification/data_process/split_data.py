import os
import random
import argparse
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("请安装 tqdm 库以使用进度条: pip install tqdm")


def split_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    将指定目录下的 CSV 文件按标签文件夹划分训练集、验证集、测试集，
    并生成对应的 txt 路径清单文件（不复制/移动原始文件）。

    Args:
        data_dir:    输入数据根目录，子文件夹名即为标签名
        output_dir:  输出目录，会生成 train.txt / val.txt / test.txt
        train_ratio: 训练集比例
        val_ratio:   验证集比例
        test_ratio:  测试集比例
        seed:        随机种子，保证可复现
    """
    # ==================== 1. 参数校验 ====================
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"比例之和必须为1，当前为 {train_ratio + val_ratio + test_ratio}"

    random.seed(seed)
    data_dir = Path(data_dir).resolve()  # 转为绝对路径，避免相对路径歧义
    output_dir = Path(output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    # ==================== 2. 收集所有 CSV 文件（带进度条） ====================
    label_files = defaultdict(list)
    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if not subdirs:
        raise ValueError(f"在 {data_dir} 下未找到任何子文件夹（标签文件夹）")

    print("📂 扫描数据目录...")
    for label_dir in tqdm(subdirs, desc="扫描标签文件夹", unit="个"):
        label = label_dir.name
        csv_files = sorted(label_dir.glob("*.csv"))
        if csv_files:
            label_files[label] = csv_files
        else:
            tqdm.write(f"⚠️  警告: 标签 '{label}' 下没有 CSV 文件，已跳过")

    if not label_files:
        raise ValueError("未找到任何 CSV 文件")

    # ==================== 3. 划分数据集 ====================
    splits = {"train": [], "val": [], "test": []}

    print("\n" + "=" * 60)
    print(f"{'标签':<20} {'总数':>6} {'训练':>6} {'验证':>6} {'测试':>6}")
    print("-" * 60)

    for label, files in sorted(label_files.items()):
        random.shuffle(files)
        n = len(files)

        n_train = max(1, int(n * train_ratio))
        n_val = max(1 if n >= 3 else 0, int(n * val_ratio))
        n_test = n - n_train - n_val

        if n_test <= 0:
            n_test = 1
            n_train = n - n_val - n_test

        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        # 存储 (绝对路径字符串, 标签名)
        splits["train"].extend([(str(f.resolve()), label) for f in train_files])
        splits["val"].extend([(str(f.resolve()), label) for f in val_files])
        splits["test"].extend([(str(f.resolve()), label) for f in test_files])

        print(f"{label:<20} {n:>6} {len(train_files):>6} {len(val_files):>6} {len(test_files):>6}")

    total_files = sum(len(v) for v in label_files.values())
    print("-" * 60)
    print(f"{'合计':<20} {total_files:>6} {len(splits['train']):>6} {len(splits['val']):>6} {len(splits['test']):>6}")
    print("=" * 60)

    # ==================== 4. 生成 txt 路径清单（带进度条） ====================
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📝 生成路径清单文件...")
    for split_name, file_list in tqdm(splits.items(), desc="写入清单", unit="个集合"):
        txt_path = output_dir / f"{split_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for file_path, label in file_list:
                # 每行格式: 文件绝对路径,标签名
                f.write(f"{file_path},{label}\n")

    print(f"\n✅ 完成！路径清单已保存到: {output_dir}")
    for split_name, file_list in splits.items():
        print(f"   {split_name}.txt: {len(file_list)} 条记录")


# ==================== 5. 命令行入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将按标签文件夹组织的 CSV 文件划分为训练/验证/测试集（生成txt路径清单）"
    )
    parser.add_argument(
        "--data_dir",
        default=r"G:\competition\2026第五届天智杯\赛前准备\雷达型号识别\test_competition_lengyan-main\test_competition_lengyan-main\datasets\synthetic_radar_overlap",
        type=str,
        help="输入数据根目录",
    )
    parser.add_argument("--output_dir", default=r"./dataset/split_data", type=str, help="输出目录（存放txt清单）")
    parser.add_argument("--train", type=float, default=0.8, help="训练集比例 (默认 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="验证集比例 (默认 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="测试集比例 (默认 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")

    args = parser.parse_args()

    split_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )