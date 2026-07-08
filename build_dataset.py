"""统一数据集构建入口。

用法:
    python build_dataset.py rts24
    python build_dataset.py case57
    python build_dataset.py --list
"""

import argparse

from tnep.config import CASES, list_cases
from tnep.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser(description="构建 TNEP 测试算例 CSV 数据集")
    parser.add_argument("case", nargs="?", choices=list_cases(), help="算例名称")
    parser.add_argument("--out-dir", default=None, help="输出目录（默认见 tnep.config.CASES）")
    parser.add_argument("--list", action="store_true", help="列出可用算例")
    args = parser.parse_args()

    if args.list:
        for name in list_cases():
            meta = CASES[name]
            print(f"  {name:10s}  {meta['label']:16s}  -> {meta['dataset_dir']}")
            print(f"             {meta['description']}")
        return

    if args.case is None:
        parser.error("请指定算例名称，或使用 --list 查看可选算例。")

    build_dataset(args.case, args.out_dir)


if __name__ == "__main__":
    main()
