"""IEEE 300-bus 投稿补强：用能覆盖 Exact 投资规模的 k 重跑启发式与四政策。

不使用 Diversity k_L=10。默认 k_L=40、k_H=12（Exact incumbent 为 19 线 / 10 站）。
每次求解立即写入 JSON，中断后可续看。
"""

from __future__ import annotations

import json
import os
import sys
import time
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


def slim(result):
    skip = {"tnep_data", "heuristic_fix_info"}
    out = {k: _jsonable(v) for k, v in result.items() if k not in skip}
    info = result.get("heuristic_fix_info") or {}
    out["pool_lines"] = _jsonable(info.get("selected_lines"))
    out["pool_storage"] = _jsonable(info.get("selected_storage"))
    return out


def dump(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


def main():
    csv_dir = dataset_dir("case300")
    out_dir = os.path.join(ROOT, "results", "case300_review_fix")
    os.makedirs(out_dir, exist_ok=True)
    time_limit = int(os.environ.get("TNEP_TL", "1800"))
    mip_gap = float(os.environ.get("TNEP_GAP", "0.001"))
    k_l = int(os.environ.get("TNEP_KL", "40"))
    k_h = int(os.environ.get("TNEP_KH", "12"))
    common = dict(
        time_limit=time_limit,
        solver_preference="gurobi",
        mip_rel_gap=mip_gap,
        gurobi_seed=2026,
        print_header=True,
    )
    print(
        f"case300 review-fix  k_L={k_l} k_H={k_h}  tl={time_limit}s  gap={mip_gap}",
        flush=True,
    )

    methods = [m.strip() for m in os.environ.get("TNEP_METHODS", "diversity,score").split(",") if m.strip()]
    method_rows = []
    for method in methods:
        name = f"heur_{method}_{k_l}_{k_h}"
        print("\n" + "=" * 60, flush=True)
        print(name, flush=True)
        t0 = time.time()
        result = solve_and_collect(
            csv_dir,
            heuristic=True,
            heuristic_method=method,
            heuristic_lines=k_l,
            heuristic_storage=k_h,
            **common,
        )
        print_result_block(name, result)
        row = slim(result)
        row["label"] = name
        row["wall_sec"] = time.time() - t0
        method_rows.append(row)
        dump(os.path.join(out_dir, f"{name}.json"), row)

    dump(os.path.join(out_dir, "method_compare.json"), {"rows": method_rows, "ts": datetime.now().isoformat()})

    if os.environ.get("TNEP_SKIP_POLICY", "0") == "1":
        print("跳过四政策（TNEP_SKIP_POLICY=1）", flush=True)
        return

    # 四政策：与主启发式同一候选池（默认 diversity）
    policy_method = methods[0]
    policies = [
        ("No-Invest", False, False),
        ("Only-Line", True, False),
        ("Only-Storage", False, True),
    ]
    policy_rows = []
    for pname, allow_line, allow_storage in policies:
        print("\n" + "=" * 60, flush=True)
        print(f"policy {pname} on {policy_method} k={k_l}/{k_h}", flush=True)
        result = solve_and_collect(
            csv_dir,
            heuristic=True,
            heuristic_method=policy_method,
            heuristic_lines=k_l,
            heuristic_storage=k_h,
            allow_line=allow_line,
            allow_storage=allow_storage,
            **common,
        )
        print_result_block(pname, result)
        row = slim(result)
        row["policy"] = pname
        policy_rows.append(row)
        dump(os.path.join(out_dir, f"policy_{pname}.json"), row)

    dump(
        os.path.join(out_dir, "policy_compare.json"),
        {"method": policy_method, "k_L": k_l, "k_H": k_h, "rows": policy_rows, "ts": datetime.now().isoformat()},
    )
    print("\n全部完成。", flush=True)


if __name__ == "__main__":
    main()
