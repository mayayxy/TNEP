"""续跑 case300：敏感性（可跳过已完成点）+ 政策对比 + 写 summary。"""

from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tnep.config import dataset_dir
from tnep.experiments import (
    DEFAULT_SENSITIVITY,
    RESULT_FIELDS,
    _serialize_result,
    ensure_results_dir,
    write_csv,
    write_json,
    write_summary_markdown,
)
from tnep.io import load_tnep_data_from_csv
from tnep.solver import print_result_block, solve_and_collect


def load_done_keys(csv_path):
    done = set()
    rows = []
    if not os.path.exists(csv_path):
        return done, rows
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            done.add((r["param_name"], str(float(r["param_mult"]))))
    return done, rows


def to_numeric_rows(raw_rows):
    out = []
    for r in raw_rows:
        item = dict(r)
        for k in (
            "obj",
            "eens_mwh",
            "curt_mwh",
            "param_value",
            "param_mult",
            "solve_time_sec",
            "mip_gap",
            "invest_line",
            "invest_stor",
            "op_cost",
            "total_storage_mwh",
        ):
            if item.get(k) not in (None, ""):
                try:
                    item[k] = float(item[k])
                except ValueError:
                    pass
        for k in ("n_lines", "n_storage"):
            if item.get(k) not in (None, ""):
                try:
                    item[k] = int(float(item[k]))
                except ValueError:
                    pass
        out.append(item)
    return out


def run_sensitivity_resume(csv_dir, results_dir, time_limit=600, mip_gap=0.01):
    ensure_results_dir(results_dir)
    sens_csv = os.path.join(results_dir, "sensitivity.csv")
    done, raw_rows = load_done_keys(sens_csv)
    rows = to_numeric_rows(raw_rows)

    base = load_tnep_data_from_csv(csv_dir)
    print("=" * 60)
    print("【case300 敏感性续跑】diversity 10/4")
    print(f"已完成点数: {len(done)}")
    print("=" * 60)

    for param_name, multipliers in DEFAULT_SENSITIVITY.items():
        base_val = float(base[param_name])
        print(f"\n--- 扫描 {param_name} (基准={base_val:,.4g}) ---")
        for mult in multipliers:
            key = (param_name, str(float(mult)))
            if key in done:
                print(f"  skip {param_name} × {mult}")
                continue
            value = base_val * float(mult)
            print(f"  {param_name} × {mult} -> {value:,.4g}")
            result = solve_and_collect(
                csv_dir,
                time_limit=time_limit,
                solver_preference="gurobi",
                mip_rel_gap=mip_gap,
                allow_line=True,
                allow_storage=True,
                param_overrides={param_name: value},
                heuristic=True,
                heuristic_method="diversity",
                heuristic_lines=10,
                heuristic_storage=4,
                print_header=False,
            )
            row = _serialize_result(
                "sensitivity",
                result,
                case="Joint",
                param_name=param_name,
                param_value=value,
                param_mult=float(mult),
            )
            rows.append(row)
            write_csv(sens_csv, rows)
            write_json(
                os.path.join(results_dir, "sensitivity.json"),
                {"experiment": "sensitivity", "rows": rows, "partial": True},
            )
            if row["obj"] is not None:
                print(
                    f"    obj={row['obj']:,.2f}, EENS={row['eens_mwh']:,.2f}, "
                    f"lines={row['n_lines']}, storage={row['n_storage']}, "
                    f"t={row['solve_time_sec']:.1f}s"
                )
            else:
                print(f"    失败: {row['termination']}")

    write_json(
        os.path.join(results_dir, "sensitivity.json"),
        {"experiment": "sensitivity", "rows": rows, "partial": False},
    )
    return rows


def run_policy(csv_dir, results_dir, time_limit=600, mip_gap=0.01):
    print("\n" + "=" * 60)
    print("【case300 政策对比】diversity 启发式")
    print("=" * 60)
    cases = [
        ("No-Invest", False, False),
        ("Only-Line", True, False),
        ("Only-Storage", False, True),
        ("Joint", True, True),
    ]
    rows = []
    for name, allow_line, allow_storage in cases:
        use_h = allow_line or allow_storage
        result = solve_and_collect(
            csv_dir,
            time_limit=time_limit,
            solver_preference="gurobi",
            mip_rel_gap=mip_gap,
            allow_line=allow_line,
            allow_storage=allow_storage,
            heuristic=use_h,
            heuristic_method="diversity",
            heuristic_lines=10,
            heuristic_storage=4,
            print_header=False,
        )
        print_result_block(name, result)
        rows.append(_serialize_result("policy_compare", result, case=name))

    write_csv(os.path.join(results_dir, "comparison_policy.csv"), rows)
    write_json(
        os.path.join(results_dir, "comparison_policy.json"),
        {"experiment": "policy_compare", "rows": rows},
    )
    return rows


def main():
    csv_dir = dataset_dir("case300")
    results_dir = os.path.join(ROOT, "results", "case300_paper")
    ensure_results_dir(results_dir)

    sens_rows = run_sensitivity_resume(csv_dir, results_dir, time_limit=600, mip_gap=0.01)
    policy_rows = run_policy(csv_dir, results_dir, time_limit=600, mip_gap=0.01)
    write_summary_markdown(
        results_dir,
        policy_rows,
        sens_rows,
        case_name="case300",
        csv_dir=csv_dir,
        notes="敏感性与政策对比均使用 diversity 启发式（10线/4储），MIPGap=1%，时限600s；完整许可证。",
    )
    print("\ncase300 实验全部完成。")


if __name__ == "__main__":
    main()
