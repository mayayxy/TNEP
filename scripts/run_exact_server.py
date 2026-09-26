"""服务器：IEEE 300-bus 全空间 Exact（无 Diversity/GA 预筛选）。

默认与论文主表同一运行层（独立充放电界，无 u_ch），便于把 25.28B incumbent 继续往下压。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tnep.config import dataset_dir
from tnep.solver import print_result_block, solve_and_collect


def _jsonable(obj):
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description="case300 全空间 Exact（服务器）")
    parser.add_argument("--case", default="case300")
    parser.add_argument("--csv-dir", default=None)
    parser.add_argument("--time-limit", type=int, default=86400, help="秒；默认 24 小时")
    parser.add_argument("--mip-gap", type=float, default=0.001, help="相对间隙，默认 0.1 percent")
    parser.add_argument("--threads", type=int, default=0, help="Gurobi 线程，0=全部可用核")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--complementarity",
        action="store_true",
        help="启用充放电互斥二元变量（与当前公式一致；主表 25.28B 未用此项）",
    )
    parser.add_argument(
        "--warmstart",
        default=None,
        help="selected_assets.json，用上次 incumbent 热启动",
    )
    args = parser.parse_args()

    csv_dir = args.csv_dir or dataset_dir(args.case)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(ROOT, "results", f"exact_server_{args.case}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    logfile = os.path.join(out_dir, "gurobi.log")
    threads = None if args.threads == 0 else args.threads

    print(f"算例: {args.case}", flush=True)
    print(f"数据: {csv_dir}", flush=True)
    print(f"输出: {out_dir}", flush=True)
    print(f"complementarity={args.complementarity}", flush=True)
    mip_start = None
    if args.warmstart:
        with open(args.warmstart, encoding="utf-8") as f:
            mip_start = json.load(f)
        print(f"热启动: {args.warmstart}", flush=True)
    tl_txt = "不限时" if not args.time_limit else f"{args.time_limit}s"
    print(f"时限: {tl_txt}  MIPGap={args.mip_gap}  threads={threads or 'auto'}", flush=True)

    result = solve_and_collect(
        csv_dir,
        time_limit=args.time_limit,
        heuristic=False,
        solver_preference="gurobi",
        gurobi_seed=args.seed,
        mip_rel_gap=args.mip_gap,
        threads=threads,
        logfile=logfile,
        complementarity=args.complementarity,
        mip_start=mip_start,
        print_header=True,
    )
    print_result_block("Exact(server)", result)

    slim = {k: _jsonable(v) for k, v in result.items() if k not in ("tnep_data", "heuristic_fix_info")}
    json_path = os.path.join(out_dir, "exact_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    lines_path = os.path.join(out_dir, "selected_assets.json")
    with open(lines_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "obj": result.get("obj"),
                "lb": result.get("lb"),
                "ub": result.get("ub"),
                "mip_gap": result.get("mip_gap"),
                "n_lines": None if result.get("selected_lines") is None else len(result["selected_lines"]),
                "n_storage": None if result.get("selected_storage") is None else len(result["selected_storage"]),
                "selected_lines": _jsonable(result.get("selected_lines")),
                "selected_storage": _jsonable(result.get("selected_storage")),
                "eens_mwh": result.get("shed_total"),
                "curt_mwh": result.get("curt_total"),
                "solve_time_sec": result.get("solve_time_sec"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {lines_path}", flush=True)
    print(f"[saved] {logfile}", flush=True)


if __name__ == "__main__":
    main()
