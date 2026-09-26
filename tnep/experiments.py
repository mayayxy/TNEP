"""对比试验与敏感性分析：结果汇总与导出。"""

import csv
import json
import os
from datetime import datetime

from tnep.heuristics import select_heuristic_candidates
from tnep.io import load_tnep_data_from_csv
from tnep.solver import make_lite_transform, run_policy_comparison, solve_and_collect


# 敏感性默认扫描点（相对基准值的倍率）
DEFAULT_SENSITIVITY = {
    "Gamma": [0.5, 0.75, 1.0, 1.25, 1.5],
    "c_shed": [0.5, 1.0, 2.0],
    "c_curt": [0.5, 1.0, 2.0, 5.0],
    "c_cap": [0.5, 0.75, 1.0, 1.25, 1.5],
}

RESULT_FIELDS = [
    "experiment",
    "case",
    "param_name",
    "param_value",
    "param_mult",
    "termination",
    "ok",
    "obj",
    "op_cost",
    "invest_line",
    "invest_stor",
    "n_lines",
    "n_storage",
    "total_storage_mwh",
    "eens_mwh",
    "curt_mwh",
    "solve_time_sec",
    "mip_gap",
]


def _serialize_result(experiment, result, case=None, param_name=None, param_value=None, param_mult=None):
    selected_lines = result.get("selected_lines")
    selected_storage = result.get("selected_storage")
    return {
        "experiment": experiment,
        "case": case if case is not None else "",
        "param_name": param_name or "",
        "param_value": param_value,
        "param_mult": param_mult,
        "termination": result.get("termination_condition"),
        "ok": bool(result.get("ok")),
        "obj": result.get("obj"),
        "op_cost": result.get("op_cost"),
        "invest_line": result.get("invest_line"),
        "invest_stor": result.get("invest_stor"),
        "n_lines": None if selected_lines is None else len(selected_lines),
        "n_storage": None if selected_storage is None else len(selected_storage),
        "total_storage_mwh": result.get("total_storage"),
        "eens_mwh": result.get("shed_total"),
        "curt_mwh": result.get("curt_total"),
        "solve_time_sec": result.get("solve_time_sec"),
        "mip_gap": result.get("mip_gap"),
        "selected_lines": selected_lines,
        "selected_storage": selected_storage,
    }


def ensure_results_dir(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in RESULT_FIELDS})


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def run_comparison_experiment(
    csv_dir,
    results_dir,
    time_limit=90,
    solver_preference="gurobi",
    mip_rel_gap=0.01,
    gurobi_seed=None,
    data_transform=None,
):
    """投资政策对比: No-Invest / Only-Line / Only-Storage / Joint。"""
    print("\n" + "=" * 60)
    print("【对比试验】投资政策四案例")
    print("=" * 60)

    raw = run_policy_comparison(
        csv_dir,
        time_limit=time_limit,
        solver_preference=solver_preference,
        mip_rel_gap=mip_rel_gap,
        gurobi_seed=gurobi_seed,
        data_transform=data_transform,
    )
    rows = [
        _serialize_result("policy_compare", r, case=r["case"])
        for r in raw
    ]

    ensure_results_dir(results_dir)
    csv_path = os.path.join(results_dir, "comparison_policy.csv")
    json_path = os.path.join(results_dir, "comparison_policy.json")
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "experiment": "policy_compare",
            "csv_dir": csv_dir,
            "time_limit": time_limit,
            "rows": rows,
        },
    )
    print(f"\n对比试验结果已保存:\n  {csv_path}\n  {json_path}")
    return rows


def run_sensitivity_experiment(
    csv_dir,
    results_dir,
    params=None,
    time_limit=90,
    solver_preference="gurobi",
    mip_rel_gap=0.01,
    gurobi_seed=None,
    heuristic=False,
    heuristic_method="score",
    heuristic_lines=10,
    heuristic_storage=4,
    data_transform=None,
):
    """单参数敏感性扫描（联合规划）。"""
    print("\n" + "=" * 60)
    print("【敏感性分析】联合规划参数扫描")
    print("=" * 60)
    if heuristic:
        print(
            f"启发式: {heuristic_method}, lines={heuristic_lines}, storage={heuristic_storage}"
        )

    base = load_tnep_data_from_csv(csv_dir)
    if data_transform is not None:
        base = data_transform(dict(base))
    sweep = params or DEFAULT_SENSITIVITY
    rows = []

    for param_name, multipliers in sweep.items():
        if param_name not in base:
            raise KeyError(f"基准数据中不存在参数 {param_name}")
        base_val = float(base[param_name])
        print(f"\n--- 扫描 {param_name} (基准={base_val:,.4g}) ---")
        for mult in multipliers:
            value = base_val * float(mult)
            print(f"  {param_name} × {mult} -> {value:,.4g}")
            result = solve_and_collect(
                csv_dir,
                time_limit=time_limit,
                solver_preference=solver_preference,
                gurobi_seed=gurobi_seed,
                allow_line=True,
                allow_storage=True,
                param_overrides={param_name: value},
                data_transform=data_transform,
                heuristic=heuristic,
                heuristic_method=heuristic_method,
                heuristic_lines=heuristic_lines,
                heuristic_storage=heuristic_storage,
                print_header=False,
                mip_rel_gap=mip_rel_gap,
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
            if row["obj"] is not None:
                print(
                    f"    obj={row['obj']:,.2f}, EENS={row['eens_mwh']:,.2f}, "
                    f"lines={row['n_lines']}, storage={row['n_storage']}, "
                    f"t={row['solve_time_sec']:.1f}s"
                )
            else:
                print(f"    失败: {row['termination']}")

            # 每点增量落盘，避免长跑中断丢结果
            ensure_results_dir(results_dir)
            csv_path = os.path.join(results_dir, "sensitivity.csv")
            json_path = os.path.join(results_dir, "sensitivity.json")
            write_csv(csv_path, rows)
            write_json(
                json_path,
                {
                    "experiment": "sensitivity",
                    "csv_dir": csv_dir,
                    "time_limit": time_limit,
                    "sweep": sweep,
                    "rows": rows,
                    "partial": True,
                },
            )

    ensure_results_dir(results_dir)
    csv_path = os.path.join(results_dir, "sensitivity.csv")
    json_path = os.path.join(results_dir, "sensitivity.json")
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "experiment": "sensitivity",
            "csv_dir": csv_dir,
            "time_limit": time_limit,
            "sweep": sweep,
            "rows": rows,
            "partial": False,
        },
    )
    print(f"\n敏感性分析结果已保存:\n  {csv_path}\n  {json_path}")
    return rows


def run_reduction_quality_experiment(
    csv_dir,
    results_dir,
    line_ks=None,
    storage_ks=None,
    exact_n_lines=None,
    exact_n_storage=None,
    exact_lines=None,
    exact_storage=None,
):
    """候选缩减质量：不求解 MILP，报告 Diversity/Score 集合相对全候选及参考 Exact 投资的覆盖。"""
    print("\n" + "=" * 60)
    print("【候选缩减质量】Diversity / Score 覆盖率")
    print("=" * 60)

    data = load_tnep_data_from_csv(csv_dir)
    n_line = len(data["candidate_lines"])
    n_stor = len(data["storage_sites"])
    line_ks = line_ks or [10, 20, 40, 80, 150, n_line]
    storage_ks = storage_ks or [4, 8, 12, 20, n_stor]
    line_ks = sorted({min(int(k), n_line) for k in line_ks})
    storage_ks = sorted({min(int(k), n_stor) for k in storage_ks})

    exact_line_set = set(tuple(x) for x in exact_lines) if exact_lines else None
    exact_stor_set = set(exact_storage) if exact_storage else None
    ref_n_lines = len(exact_line_set) if exact_line_set is not None else exact_n_lines
    ref_n_stor = len(exact_stor_set) if exact_stor_set is not None else exact_n_storage

    rows = []
    for k_l in line_ks:
        for k_h in storage_ks:
            div_l_list, div_h_list = select_heuristic_candidates(
                data, max_lines=k_l, max_storage=k_h, method="diversity"
            )
            sco_l_list, sco_h_list = select_heuristic_candidates(
                data, max_lines=k_l, max_storage=k_h, method="score"
            )
            div_l, div_h = set(div_l_list), set(div_h_list)
            sco_l, sco_h = set(sco_l_list), set(sco_h_list)
            row = {
                "k_L": k_l,
                "k_H": k_h,
                "n_candidate_lines": n_line,
                "n_storage_sites": n_stor,
                "line_jaccard_div_score": (
                    len(div_l & sco_l) / max(len(div_l | sco_l), 1)
                ),
                "storage_jaccard_div_score": (
                    len(div_h & sco_h) / max(len(div_h | sco_h), 1)
                ),
                "exact_line_recall_cap": (
                    None if not ref_n_lines else min(k_l, ref_n_lines) / float(ref_n_lines)
                ),
                "exact_storage_recall_cap": (
                    None if not ref_n_stor else min(k_h, ref_n_stor) / float(ref_n_stor)
                ),
                "exact_line_recall": (
                    None
                    if exact_line_set is None
                    else len(div_l & exact_line_set) / max(len(exact_line_set), 1)
                ),
                "exact_storage_recall": (
                    None
                    if exact_stor_set is None
                    else len(div_h & exact_stor_set) / max(len(exact_stor_set), 1)
                ),
            }
            rows.append(row)
            print(
                f"  k_L={k_l:3d} k_H={k_h:2d}  "
                f"Jacc_L={row['line_jaccard_div_score']:.3f}  "
                f"Jacc_H={row['storage_jaccard_div_score']:.3f}  "
                f"RecallCap_L={row['exact_line_recall_cap']}  "
                f"RecallCap_H={row['exact_storage_recall_cap']}"
            )

    payload = {
        "experiment": "reduction_quality",
        "csv_dir": csv_dir,
        "n_candidate_lines": n_line,
        "n_storage_sites": n_stor,
        "exact_n_lines": ref_n_lines,
        "exact_n_storage": ref_n_stor,
        "rows": rows,
    }
    ensure_results_dir(results_dir)
    json_path = os.path.join(results_dir, "reduction_quality.json")
    csv_path = os.path.join(results_dir, "reduction_quality.csv")
    write_json(json_path, payload)
    fieldnames = [
        "k_L",
        "k_H",
        "n_candidate_lines",
        "n_storage_sites",
        "line_jaccard_div_score",
        "storage_jaccard_div_score",
        "exact_line_recall_cap",
        "exact_storage_recall_cap",
        "exact_line_recall",
        "exact_storage_recall",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"候选缩减质量已保存:\n  {csv_path}\n  {json_path}")
    return rows


def write_summary_markdown(results_dir, comparison_rows, sensitivity_rows, case_name, csv_dir, notes=None):
    path = os.path.join(results_dir, "summary.md")
    lines = [
        f"# 实验结果摘要 ({case_name})",
        "",
        f"- 数据目录: `{csv_dir}`",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if notes:
        lines.append(f"- 备注: {notes}")
    lines.extend(
        [
            "",
            "## 1. 投资政策对比",
            "",
            "| Case | Obj (USD/yr) | Invest Line | Invest Storage | EENS (MWh/yr) | Curt (MWh/yr) | #Lines | #Storage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in comparison_rows:
        def fmt(x, nd=2):
            return "N/A" if x is None else f"{x:,.{nd}f}"

        lines.append(
            f"| {r['case']} | {fmt(r['obj'])} | {fmt(r['invest_line'])} | {fmt(r['invest_stor'])} | "
            f"{fmt(r['eens_mwh'])} | {fmt(r['curt_mwh'])} | {r['n_lines']} | {r['n_storage']} |"
        )

    lines.extend(["", "## 2. 敏感性分析", ""])
    by_param = {}
    for r in sensitivity_rows:
        by_param.setdefault(r["param_name"], []).append(r)

    for param_name, group in by_param.items():
        lines.extend(
            [
                f"### {param_name}",
                "",
                "| Mult | Value | Obj | EENS | Curt | #Lines | #Storage | Storage MWh |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for r in group:
            def fmt(x, nd=2):
                return "N/A" if x is None else f"{x:,.{nd}f}"

            lines.append(
                f"| {r['param_mult']} | {fmt(r['param_value'], 4)} | {fmt(r['obj'])} | "
                f"{fmt(r['eens_mwh'])} | {fmt(r['curt_mwh'])} | {r['n_lines']} | "
                f"{r['n_storage']} | {fmt(r['total_storage_mwh'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 3. 文件",
            "",
            "- `comparison_policy.csv` / `.json`",
            "- `sensitivity.csv` / `.json`",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"摘要已保存: {path}")
    return path
