"""依次跑 RTS-GMLC 全模型实验 + case300 diversity 敏感性。"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    rts_out = os.path.join(ROOT, "results", "rts_gmlc_paper")
    c300_out = os.path.join(ROOT, "results", "case300_paper")

    # 1) RTS-GMLC: 政策对比 + 敏感性（精确，无 heuristic）
    run(
        [
            PY,
            "-u",
            "run_experiments.py",
            "--case",
            "rts_gmlc",
            "--only",
            "all",
            "--time-limit",
            "600",
            "--mip-gap",
            "0.01",
            "--solver",
            "gurobi",
            "--out-dir",
            rts_out,
        ]
    )

    # 2) case300: 敏感性用 diversity 启发式（政策对比也跑，但用启发式加速）
    # 政策对比暂用较短时限+启发式不支持四案例内部，故只跑 sensitivity；
    # 另跑一次带 heuristic 的 joint 四点可手写，这里先完整敏感性。
    run(
        [
            PY,
            "-u",
            "run_experiments.py",
            "--case",
            "case300",
            "--only",
            "sensitivity",
            "--time-limit",
            "600",
            "--mip-gap",
            "0.01",
            "--solver",
            "gurobi",
            "--heuristic",
            "--heuristic-method",
            "diversity",
            "--heuristic-lines",
            "10",
            "--heuristic-storage",
            "4",
            "--out-dir",
            c300_out,
        ]
    )

    # case300 政策对比：用 diversity + 600s（通过 solve_tnep policy-compare 不支持 heuristic）
    # 用精确四案例在 600s 可能无解；改为手动四案例+启发式脚本内嵌调用
    from tnep.experiments import run_comparison_experiment, write_summary_markdown
    import csv
    import json

    # 对 case300 用启发式跑四政策：临时扩展
    from tnep.solver import solve_and_collect, print_result_block

    print("\n" + "=" * 60)
    print("【case300 政策对比】diversity 启发式")
    print("=" * 60)
    os.makedirs(c300_out, exist_ok=True)
    cases = [
        ("No-Invest", False, False),
        ("Only-Line", True, False),
        ("Only-Storage", False, True),
        ("Joint", True, True),
    ]
    rows = []
    for name, allow_line, allow_storage in cases:
        # No-Invest 无需 heuristic；有投资时用 diversity 缩候选
        use_h = allow_line or allow_storage
        result = solve_and_collect(
            os.path.join(ROOT, "dataset_csv_case300"),
            time_limit=600,
            solver_preference="gurobi",
            mip_rel_gap=0.01,
            allow_line=allow_line,
            allow_storage=allow_storage,
            heuristic=use_h,
            heuristic_method="diversity",
            heuristic_lines=10,
            heuristic_storage=4,
            print_header=False,
        )
        print_result_block(name, result)
        from tnep.experiments import _serialize_result

        rows.append(_serialize_result("policy_compare", result, case=name))

    csv_path = os.path.join(c300_out, "comparison_policy.csv")
    json_path = os.path.join(c300_out, "comparison_policy.json")
    from tnep.experiments import RESULT_FIELDS, write_csv, write_json

    write_csv(csv_path, rows)
    write_json(json_path, {"experiment": "policy_compare", "rows": rows})

    # 合并 summary
    sens_path = os.path.join(c300_out, "sensitivity.csv")
    sens_rows = []
    if os.path.exists(sens_path):
        with open(sens_path, "r", encoding="utf-8-sig") as f:
            sens_rows = list(csv.DictReader(f))
            for r in sens_rows:
                for k in ("obj", "eens_mwh", "curt_mwh", "param_value", "param_mult", "solve_time_sec", "mip_gap"):
                    if r.get(k) not in (None, ""):
                        try:
                            r[k] = float(r[k])
                        except ValueError:
                            pass
                for k in ("n_lines", "n_storage"):
                    if r.get(k) not in (None, ""):
                        try:
                            r[k] = int(float(r[k]))
                        except ValueError:
                            pass

    write_summary_markdown(
        c300_out,
        rows,
        sens_rows,
        case_name="case300",
        csv_dir=os.path.join(ROOT, "dataset_csv_case300"),
        notes="敏感性与政策对比均使用 diversity 启发式（10线/4储），MIPGap=1%，时限600s；完整许可证。",
    )
    print("\n全部流水线完成。")


if __name__ == "__main__":
    main()
