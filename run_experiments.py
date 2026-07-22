"""运行对比试验与敏感性分析。

用法:
    python run_experiments.py --case rts24
    python run_experiments.py --case rts24 --only comparison
    python run_experiments.py --case rts24 --only sensitivity --time-limit 60
"""

import argparse
import os
from datetime import datetime

from tnep.config import CASES, dataset_dir, list_cases
from tnep.experiments import (
    DEFAULT_SENSITIVITY,
    run_comparison_experiment,
    run_sensitivity_experiment,
    write_summary_markdown,
)
from tnep.solver import make_lite_transform


def main():
    parser = argparse.ArgumentParser(description="TNEP 对比试验与敏感性分析")
    parser.add_argument("--case", choices=list_cases(), default="rts24", help="测试算例")
    parser.add_argument("--csv-dir", default=None, help="自定义数据目录（覆盖 --case）")
    parser.add_argument(
        "--only",
        choices=["all", "comparison", "sensitivity"],
        default="all",
        help="只跑对比 / 只跑敏感性 / 全部",
    )
    parser.add_argument("--time-limit", type=int, default=90, help="单次求解时间上限（秒）")
    parser.add_argument("--mip-gap", type=float, default=0.01, help="MIP 相对 gap")
    parser.add_argument("--solver", choices=["gurobi", "auto"], default="gurobi")
    parser.add_argument("--gurobi-seed", type=int, default=2026)
    parser.add_argument("--heuristic", action="store_true", help="敏感性分析时启用启发式")
    parser.add_argument(
        "--lite",
        action="store_true",
        help="精简模型（减少场景/时段/候选），适配 Gurobi 受限许可证",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="结果输出目录（默认 results/<case>_<timestamp>）",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：敏感性参数点更少，适合冒烟测试",
    )
    args = parser.parse_args()

    csv_dir = args.csv_dir or dataset_dir(args.case)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"{args.case}_{stamp}",
    )
    os.makedirs(results_dir, exist_ok=True)

    data_transform = make_lite_transform() if args.lite else None

    print(f"算例: {args.case} ({CASES[args.case]['label']})")
    print(f"数据: {csv_dir}")
    print(f"输出: {results_dir}")
    if args.lite:
        print("模式: lite（受限许可证精简模型）")

    common = {
        "time_limit": args.time_limit,
        "solver_preference": args.solver,
        "mip_rel_gap": args.mip_gap,
        "gurobi_seed": args.gurobi_seed,
        "data_transform": data_transform,
    }

    comparison_rows = []
    sensitivity_rows = []

    if args.only in ("all", "comparison"):
        comparison_rows = run_comparison_experiment(csv_dir, results_dir, **common)

    if args.only in ("all", "sensitivity"):
        if args.quick:
            sweep = {
                "Gamma": [0.75, 1.0, 1.25],
                "c_shed": [0.5, 1.0, 2.0],
                "c_curt": [1.0, 2.0],
                "c_cap": [0.75, 1.0, 1.25],
            }
        else:
            sweep = DEFAULT_SENSITIVITY
        sensitivity_rows = run_sensitivity_experiment(
            csv_dir,
            results_dir,
            params=sweep,
            heuristic=args.heuristic,
            **common,
        )

    if comparison_rows or sensitivity_rows:
        notes = None
        if args.lite:
            notes = (
                "lite 模式：场景/时段/候选线路已压缩，适配 Gurobi 受限许可证；"
                "正式论文结果请使用学术/商业许可证并去掉 --lite"
            )
        write_summary_markdown(
            results_dir,
            comparison_rows,
            sensitivity_rows,
            case_name=args.case,
            csv_dir=csv_dir,
            notes=notes,
        )

    print("\n全部实验完成。")


if __name__ == "__main__":
    main()
