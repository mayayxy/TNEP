"""TNEP + 储能联合规划统一求解入口。

用法:
    python solve_tnep.py --case rts24 --policy-compare
    python solve_tnep.py --case case57
    python solve_tnep.py --case rts_gmlc --heuristic
    python solve_tnep.py --csv-dir dataset_csv_case300 --compare
"""

import argparse

from tnep.config import CASES, dataset_dir, list_cases
from tnep.solver import run_policy_comparison, solve_compare, solve_once


def resolve_csv_dir(args):
    if args.csv_dir:
        return args.csv_dir
    if args.case:
        return dataset_dir(args.case)
    raise SystemExit("请指定 --case 或 --csv-dir。运行 python build_dataset.py --list 查看算例。")


def main():
    parser = argparse.ArgumentParser(description="TNEP + 储能联合规划 MILP 求解")
    parser.add_argument("--case", choices=list_cases(), help="测试算例（自动映射到 dataset_csv_* 目录）")
    parser.add_argument("--csv-dir", default=None, help="数据目录（覆盖 --case）")
    parser.add_argument("--time-limit", type=int, default=120, help="求解时间上限（秒）；0 表示不设上限")
    parser.add_argument("--policy-compare", action="store_true", help="四案例对比: 不投资/只扩网/只建储/联合")
    parser.add_argument("--compare", action="store_true", help="精确求解 vs 启发式求解对比")
    parser.add_argument("--heuristic", action="store_true", help="启用启发式预筛选")
    parser.add_argument("--strong-heuristic", action="store_true", help="强启发式：固定选中投资")
    parser.add_argument("--exact-time-limit", type=int, default=None, help="compare 模式下精确求解时间上限")
    parser.add_argument("--heuristic-time-limit", type=int, default=None, help="compare 模式下启发式求解时间上限")
    parser.add_argument("--heuristic-lines", type=int, default=10, help="启发式保留候选线路数")
    parser.add_argument("--heuristic-storage", type=int, default=4, help="启发式保留储能候选数")
    parser.add_argument(
        "--heuristic-method",
        choices=["score", "bridge", "budget_greedy", "diversity"],
        default="score",
        help="启发式策略",
    )
    parser.add_argument("--solver", choices=["gurobi", "auto"], default="gurobi", help="求解器（默认 Gurobi，auto 同 gurobi）")
    parser.add_argument("--gurobi-seed", type=int, default=None, help="Gurobi 随机种子")
    args = parser.parse_args()

    csv_dir = resolve_csv_dir(args)
    if args.case:
        print(f"算例: {args.case} ({CASES[args.case]['label']})")

    common = {
        "heuristic_lines": args.heuristic_lines,
        "heuristic_storage": args.heuristic_storage,
        "heuristic_method": args.heuristic_method,
        "solver_preference": args.solver,
        "gurobi_seed": args.gurobi_seed,
    }

    if args.policy_compare:
        run_policy_comparison(csv_dir, time_limit=args.time_limit, solver_preference=args.solver)
    elif args.compare:
        solve_compare(
            csv_dir,
            time_limit=args.time_limit,
            exact_time_limit=args.exact_time_limit,
            heuristic_time_limit=args.heuristic_time_limit,
            strong_heuristic=args.strong_heuristic,
            **common,
        )
    else:
        solve_once(
            csv_dir,
            time_limit=args.time_limit,
            heuristic=args.heuristic,
            strong_heuristic=args.strong_heuristic,
            **common,
        )


if __name__ == "__main__":
    main()
