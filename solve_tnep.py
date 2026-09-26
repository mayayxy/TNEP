"""TNEP + 储能联合规划统一求解入口。

用法:
    python solve_tnep.py --case rts24 --policy-compare
    python solve_tnep.py --case case57
    python solve_tnep.py --case rts_gmlc --heuristic
    python solve_tnep.py --csv-dir dataset_csv_case300 --compare
"""

import argparse
import os

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
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.01,
        help="MIP 相对 gap 容差（默认 0.01=1%%；精确求解可设 0.001 或 0）",
    )
    parser.add_argument("--alns", action="store_true", help="启用 ALNS 搜索投资决策")
    parser.add_argument("--alns-time-limit", type=int, default=1200, help="ALNS 阶段时限（秒）")
    parser.add_argument("--alns-then-exact", action="store_true", help="(默认) ALNS 后热启动精确求解；与 --alns-only 互斥")
    parser.add_argument("--alns-only", action="store_true", help="只跑 ALNS，不启动精确求解")
    parser.add_argument("--alns-pool-lines", type=int, default=120, help="ALNS/GA/Tabu 候选线路池大小")
    parser.add_argument("--alns-pool-storage", type=int, default=20, help="ALNS/GA/Tabu 候选储能池大小")
    parser.add_argument(
        "--alns-pool-method",
        choices=["score", "bridge", "budget_greedy", "diversity", "shed_union", "all"],
        default="all",
        help="候选池: all=全部候选; shed_union=评分∪切负荷走廊",
    )
    parser.add_argument("--alns-max-lines", type=int, default=25, help="ALNS 解中最多线路数")
    parser.add_argument("--alns-max-storage", type=int, default=12, help="ALNS 解中最多储能站数")
    parser.add_argument("--alns-eval-time", type=int, default=60, help="ALNS 每次邻域评估时限（秒）")
    parser.add_argument("--alns-seed", type=int, default=1, help="ALNS 随机种子")
    parser.add_argument("--ga", action="store_true", help="遗传算法搜索投资决策")
    parser.add_argument("--tabu", action="store_true", help="禁忌搜索投资决策")
    parser.add_argument("--meta-time-limit", type=int, default=1200, help="GA/Tabu 各自搜索时限（秒）")
    parser.add_argument("--meta-eval-time", type=int, default=90, help="GA/Tabu 每次适应度评估时限（秒）")
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
        "mip_rel_gap": args.mip_gap,
    }

    if args.ga or args.tabu:
        from tnep.metaheuristics import run_ga_tabu_pipeline

        methods = []
        if args.ga:
            methods.append("ga")
        if args.tabu:
            methods.append("tabu")
        run_ga_tabu_pipeline(
            csv_dir,
            methods=tuple(methods),
            search_time=args.meta_time_limit,
            eval_time_limit=args.meta_eval_time,
            pool_lines=args.alns_pool_lines,
            pool_storage=args.alns_pool_storage,
            pool_method=args.alns_pool_method,
            max_lines=args.alns_max_lines,
            max_storage=args.alns_max_storage,
            seed=args.alns_seed,
            solver_preference=args.solver,
            out_dir=os.path.join("results", "case300_metaheuristics_v2") if args.case == "case300" else os.path.join("results", "metaheuristics"),
        )
    elif args.alns:
        from tnep.alns import solve_alns_pipeline

        exact_tl = args.exact_time_limit if args.exact_time_limit is not None else args.time_limit
        solve_alns_pipeline(
            csv_dir,
            alns_time_limit=args.alns_time_limit,
            exact_time_limit=exact_tl,
            mip_rel_gap=args.mip_gap,
            pool_lines=args.alns_pool_lines,
            pool_storage=args.alns_pool_storage,
            pool_method=args.alns_pool_method,
            max_lines=args.alns_max_lines,
            max_storage=args.alns_max_storage,
            seed=args.alns_seed,
            solver_preference=args.solver,
            then_exact=not args.alns_only,
            eval_time_limit=args.alns_eval_time,
        )
    elif args.policy_compare:
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
