"""MILP 求解、结果汇总与对比实验。"""

import math
import sys
import time
from collections import defaultdict

from pyomo.environ import SolverFactory, value

from tnep.heuristics import apply_heuristic_fixings
from tnep.model import build_model, shrink_tnep_data


def pick_solver(preferred="gurobi"):
    solver_name = "gurobi" if preferred in (None, "auto") else preferred
    if SolverFactory(solver_name).available(False):
        return solver_name
    py = sys.executable
    if solver_name == "gurobi":
        raise RuntimeError(
            "Gurobi 求解器不可用。"
            f"\n当前 Python: {py}"
            "\n请确认已安装 gurobipy 且许可证有效（GRB_LICENSE_FILE 或默认路径）。"
        )
    raise RuntimeError(f"指定求解器不可用: {solver_name}")


def normalize_time_limit(time_limit):
    if time_limit is None:
        return None
    try:
        tl = float(time_limit)
    except (TypeError, ValueError):
        return None
    if tl <= 0:
        return None
    return tl


def configure_solver(solver_name, time_limit=120, mip_rel_gap=0.01, gurobi_seed=None, threads=None, logfile=None):
    solver = SolverFactory(solver_name)
    normalized_tl = normalize_time_limit(time_limit)
    if hasattr(solver, "config"):
        try:
            solver.config.time_limit = normalized_tl
            solver.config.rel_gap = mip_rel_gap
            solver.config.raise_exception_on_nonoptimal_result = False
        except Exception:
            pass
    if solver_name in ("appsi_highs", "highs"):
        opts = solver.highs_options if solver_name == "appsi_highs" else solver.options
        if normalized_tl is not None:
            opts["time_limit"] = normalized_tl
        elif solver_name == "highs" and "time_limit" in opts:
            del opts["time_limit"]
        opts["mip_rel_gap"] = mip_rel_gap
    elif solver_name == "gurobi":
        if normalized_tl is not None:
            solver.options["TimeLimit"] = normalized_tl
        elif "TimeLimit" in solver.options:
            del solver.options["TimeLimit"]
        solver.options["MIPGap"] = mip_rel_gap
        if gurobi_seed is not None:
            solver.options["Seed"] = int(gurobi_seed)
        if threads is not None:
            solver.options["Threads"] = int(threads)
        if logfile:
            solver.options["LogFile"] = str(logfile)
            solver.options["LogToConsole"] = 1
    return solver


def extract_bounds_and_gap(result):
    try:
        p = result["Problem"][0]
        lb = float(p["Lower bound"])
        ub = float(p["Upper bound"])
        if math.isfinite(lb) and math.isfinite(ub):
            gap = abs(ub - lb) / max(abs(ub), 1e-9)
            return lb, ub, gap
        return lb, ub, None
    except Exception:
        return None, None, None


def apply_investment_policy(model, allow_line=True, allow_storage=True):
    if not allow_line:
        for l in model.LC:
            model.y[l].fix(0)
    if not allow_storage:
        for h in model.H:
            model.z[h].fix(0)
            model.E[h].fix(0)


def fallback_feasible_solution_metrics(tnep_data):
    """无扩建、无储能、零潮流的本地平衡基线。"""
    buses = tnep_data["buses"]
    omega = tnep_data["omega"]
    delta_t = tnep_data["delta_t"]
    c_shed = tnep_data["c_shed"]
    c_curt = tnep_data["c_curt"]

    gen_at_bus = defaultdict(list)
    for g, meta in tnep_data["gen_meta"].items():
        gen_at_bus[meta["bus"]].append((g, meta["pmax"], tnep_data["c_gen"][g]))
    for b in gen_at_bus:
        gen_at_bus[b].sort(key=lambda x: x[2])

    ren_at_bus = defaultdict(list)
    for rid, meta in tnep_data["renewable_meta"].items():
        ren_at_bus[meta["bus"]].append(rid)

    shed_total = 0.0
    curt_total = 0.0
    gen_cost_total = 0.0

    for s in tnep_data["scenarios"]:
        w_days = omega[s]
        for t in tnep_data["periods"]:
            for b in buses:
                demand_bt = tnep_data["demand"][(b, s, t)]
                ren_avail_bt = sum(tnep_data["wbar"][(rid, s, t)] for rid in ren_at_bus[b])
                ren_used = min(demand_bt, ren_avail_bt)
                remain = demand_bt - ren_used
                curt_bt = ren_avail_bt - ren_used

                gen_cost_bt = 0.0
                if remain > 0:
                    for _, pmax, c_gen in gen_at_bus.get(b, []):
                        if remain <= 0:
                            break
                        take = min(remain, pmax)
                        gen_cost_bt += take * c_gen
                        remain -= take

                shed_bt = max(0.0, remain)
                shed_total += w_days * shed_bt * delta_t
                curt_total += w_days * curt_bt * delta_t
                gen_cost_total += w_days * gen_cost_bt * delta_t

    op_cost = gen_cost_total + c_shed * shed_total + c_curt * curt_total
    return {
        "obj": op_cost,
        "invest_line": 0.0,
        "invest_stor": 0.0,
        "selected_lines": [],
        "selected_storage": [],
        "total_storage": 0.0,
        "shed_total": shed_total,
        "curt_total": curt_total,
    }


def collect_solution_metrics(model):
    selected_lines = [l for l in model.LC if value(model.y[l]) > 0.5]
    selected_storage = [h for h in model.H if value(model.z[h]) > 0.5]
    total_storage = sum(value(model.E[h]) for h in model.H)
    invest_line = sum(value(model.c_line[l]) * value(model.y[l]) for l in model.LC)
    invest_stor = sum(
        value(model.c_fix[h]) * value(model.z[h]) + value(model.c_cap) * value(model.E[h]) for h in model.H
    )
    shed_total = sum(
        value(model.omega[s])
        * sum(value(model.d_shed[b, s, t]) * value(model.delta_t) for b in model.B for t in model.T)
        for s in model.S
    )
    curt_total = sum(
        value(model.omega[s])
        * sum(
            (value(model.Wbar[r, s, t]) - value(model.w[r, s, t])) * value(model.delta_t)
            for r in model.R
            for t in model.T
        )
        for s in model.S
    )
    total_obj = value(model.obj)
    return {
        "obj": total_obj,
        "op_cost": total_obj - invest_line - invest_stor,
        "invest_line": invest_line,
        "invest_stor": invest_stor,
        "selected_lines": selected_lines,
        "selected_storage": selected_storage,
        "total_storage": total_storage,
        "shed_total": shed_total,
        "curt_total": curt_total,
        "gamma": value(model.Gamma),
    }


def solve_and_collect(
    csv_dir,
    time_limit=120,
    heuristic=False,
    heuristic_lines=10,
    heuristic_storage=4,
    strong_heuristic=False,
    heuristic_method="score",
    solver_preference="gurobi",
    gurobi_seed=None,
    allow_line=True,
    allow_storage=True,
    param_overrides=None,
    data_transform=None,
    print_header=True,
    mip_rel_gap=0.01,
    threads=None,
    logfile=None,
    complementarity=True,
    mip_start=None,
):
    model, tnep_data = build_model(
        csv_dir,
        param_overrides=param_overrides,
        data_transform=data_transform,
        complementarity=complementarity,
    )
    apply_investment_policy(model, allow_line=allow_line, allow_storage=allow_storage)

    solver_name = pick_solver(preferred=solver_preference)

    if print_header:
        print(f"数据目录: {csv_dir}")
        print(f"求解器: {solver_name}")

    heuristic_fix_info = None
    if heuristic:
        heuristic_fix_info = apply_heuristic_fixings(
            model,
            tnep_data,
            heuristic_lines=heuristic_lines,
            heuristic_storage=heuristic_storage,
            strong_heuristic=strong_heuristic,
            heuristic_method=heuristic_method,
        )
        if print_header:
            print(
                f"启发式已启用: 线路 {len(heuristic_fix_info['selected_lines'])}/{len(list(model.LC))}, "
                f"储能 {len(heuristic_fix_info['selected_storage'])}/{len(list(model.H))} "
                f"({heuristic_method})"
            )

    use_warmstart = False
    if mip_start:
        ls = {tuple(x) if not isinstance(x, tuple) else x for x in (mip_start.get("selected_lines") or [])}
        ss = set(mip_start.get("selected_storage") or [])
        for l in model.LC:
            model.y[l].set_value(1.0 if l in ls else 0.0)
        for h in model.H:
            model.z[h].set_value(1.0 if h in ss else 0.0)
            if h not in ss:
                model.E[h].set_value(0.0)
        use_warmstart = True
        if print_header:
            print(f"MIP start: {len(ls)} 条线路, {len(ss)} 个储能站", flush=True)

    result = None
    solve_time_sec = None
    last_exc = None

    solver = configure_solver(
        solver_name,
        time_limit=time_limit,
        mip_rel_gap=mip_rel_gap,
        gurobi_seed=gurobi_seed,
        threads=threads,
        logfile=logfile,
    )
    t0 = time.perf_counter()
    try:
        result = solver.solve(model, load_solutions=False, warmstart=use_warmstart)
        solve_time_sec = time.perf_counter() - t0
    except Exception as e:
        solve_time_sec = time.perf_counter() - t0
        last_exc = e
        result = None

    if result is None:
        e = last_exc if last_exc is not None else RuntimeError("Gurobi 求解失败")
        return {
            "termination_condition": f"FAILED: {type(e).__name__}: {e}",
            "solve_time_sec": solve_time_sec or 0.0,
            "lb": None,
            "ub": None,
            "mip_gap": None,
            "ok": False,
            "solution_source": "solver_exception",
            "tnep_data": tnep_data,
            "heuristic_fix_info": heuristic_fix_info,
            **{k: None for k in ("obj", "selected_lines", "selected_storage", "total_storage", "invest_line", "invest_stor", "shed_total", "curt_total", "gamma", "op_cost")},
        }

    lb, ub, mip_gap = extract_bounds_and_gap(result)
    solution_count = len(result.solution) if hasattr(result, "solution") else 0

    if solution_count <= 0:
        fallback = fallback_feasible_solution_metrics(tnep_data)
        return {
            "termination_condition": f"{result.solver.termination_condition} (no incumbent, fallback)",
            "solve_time_sec": solve_time_sec,
            "lb": lb,
            "ub": ub,
            "mip_gap": mip_gap,
            "ok": True,
            "solution_source": "fallback_feasible_dispatch",
            "tnep_data": tnep_data,
            "heuristic_fix_info": heuristic_fix_info,
            "op_cost": fallback["obj"],
            **fallback,
        }

    try:
        model.solutions.load_from(result)
    except Exception as e:
        return {
            "termination_condition": f"{result.solver.termination_condition} (solution load failed: {e})",
            "solve_time_sec": solve_time_sec,
            "lb": lb,
            "ub": ub,
            "mip_gap": mip_gap,
            "ok": False,
            "solution_source": "solution_load_failed",
            "tnep_data": tnep_data,
            "heuristic_fix_info": heuristic_fix_info,
            **{k: None for k in ("obj", "selected_lines", "selected_storage", "total_storage", "invest_line", "invest_stor", "shed_total", "curt_total", "gamma", "op_cost")},
        }

    metrics = collect_solution_metrics(model)
    return {
        "termination_condition": str(result.solver.termination_condition),
        "solve_time_sec": solve_time_sec,
        "lb": lb,
        "ub": ub,
        "mip_gap": mip_gap,
        "ok": True,
        "solution_source": "incumbent",
        "tnep_data": tnep_data,
        "heuristic_fix_info": heuristic_fix_info,
        **metrics,
    }


def print_result_block(name, result):
    print(f"\n=== {name} ===")
    print(f"终止状态: {result['termination_condition']}")
    print(f"结果来源: {result.get('solution_source', 'unknown')}")
    print(f"求解时间: {result['solve_time_sec']:.2f} 秒")
    if result["lb"] is not None and result["ub"] is not None:
        print(f"下界/上界: {result['lb']:,.2f} / {result['ub']:,.2f}")
    if result["mip_gap"] is not None:
        print(f"MIP GAP: {result['mip_gap']:.4%}")
    if result["obj"] is None:
        print("目标函数值: N/A")
    else:
        print(f"目标函数值: {result['obj']:,.2f}")
        if result.get("op_cost") is not None:
            print(f"运行成本(估算): {result['op_cost']:,.2f} USD/year")
    if result["selected_lines"] is not None:
        print(f"已选线路: {len(result['selected_lines'])}")
    if result["selected_storage"] is not None:
        print(f"已选储能站点: {len(result['selected_storage'])}")
    if result["total_storage"] is not None:
        print(f"储能容量: {result['total_storage']:.2f} MWh")
    if result["invest_line"] is not None:
        print(f"线路投资: {result['invest_line']:,.2f} USD/year")
    if result["invest_stor"] is not None:
        print(f"储能投资: {result['invest_stor']:,.2f} USD/year")
    if result["shed_total"] is not None:
        print(f"EENS: {result['shed_total']:,.2f} MWh/year")
    if result["curt_total"] is not None:
        print(f"弃电量: {result['curt_total']:,.2f} MWh/year")


def solve_once(csv_dir, **kwargs):
    result = solve_and_collect(csv_dir, print_header=True, **kwargs)
    print_result_block("求解结果", result)
    return result


def solve_compare(csv_dir, time_limit=120, exact_time_limit=None, heuristic_time_limit=None, **kwargs):
    exact_limit = normalize_time_limit(exact_time_limit if exact_time_limit is not None else time_limit)
    heur_limit = normalize_time_limit(heuristic_time_limit if heuristic_time_limit is not None else time_limit)

    print(f"数据目录: {csv_dir}")
    print("对比: 精确求解 vs 启发式求解")
    exact = solve_and_collect(csv_dir, time_limit=exact_limit, heuristic=False, print_header=False, **kwargs)
    heur = solve_and_collect(
        csv_dir,
        time_limit=heur_limit,
        heuristic=True,
        print_header=False,
        **kwargs,
    )
    print_result_block("精确求解", exact)
    print_result_block("启发式求解", heur)

    print("\n=== 对比结论 ===")
    if exact["obj"] is not None and heur["obj"] is not None:
        diff = heur["obj"] - exact["obj"]
        print(f"目标值差异: {diff:,.2f} ({diff / max(abs(exact['obj']), 1e-9):.4%})")
    print(f"耗时比: {exact['solve_time_sec'] / max(heur['solve_time_sec'], 1e-9):.2f}x")
    return exact, heur


def run_policy_comparison(
    csv_dir,
    time_limit=90,
    solver_preference="gurobi",
    mip_rel_gap=0.005,
    param_overrides=None,
    data_transform=None,
    gurobi_seed=None,
):
    """四案例对比: No-Invest / Only-Line / Only-Storage / Joint。"""
    print(f"使用求解器: {pick_solver(preferred=solver_preference)}")
    print("\n开始对比实验: No-Invest / Only-Line / Only-Storage / Joint")

    cases = [
        ("No-Invest", False, False),
        ("Only-Line", True, False),
        ("Only-Storage", False, True),
        ("Joint", True, True),
    ]
    results = []
    for case_name, allow_line, allow_storage in cases:
        result = solve_and_collect(
            csv_dir,
            time_limit=time_limit,
            solver_preference=solver_preference,
            gurobi_seed=gurobi_seed,
            allow_line=allow_line,
            allow_storage=allow_storage,
            param_overrides=param_overrides,
            data_transform=data_transform,
            print_header=False,
            mip_rel_gap=mip_rel_gap,
        )
        print(f"\n=== {case_name} ===")
        print(f"终止状态: {result['termination_condition']}")
        if result["obj"] is not None:
            print(f"目标函数值: {result['obj']:,.2f}")
            print(f"EENS: {result['shed_total']:,.2f} MWh/year")
            print(f"弃电: {result['curt_total']:,.2f} MWh/year")
            n_line = len(result["selected_lines"] or [])
            n_stor = len(result["selected_storage"] or [])
            print(f"已选线路: {n_line}, 已选储能: {n_stor}")
        results.append({"case": case_name, **result})

    baseline = next((r for r in results if r["case"] == "No-Invest"), results[0])
    print("\n=== 对比汇总 (相对 No-Invest) ===")
    print("Case | Obj(USD/yr) | Delta Obj | EENS | Delta EENS | Curt")
    for r in results:
        if r["obj"] is None or baseline.get("obj") is None:
            print(f"{r['case']} | N/A")
            continue
        print(
            f"{r['case']} | {r['obj']:,.2f} | {r['obj'] - baseline['obj']:,.2f} | "
            f"{r['shed_total']:,.2f} | {r['shed_total'] - baseline['shed_total']:,.2f} | "
            f"{r['curt_total']:,.2f}"
        )
    return results


def make_lite_transform(
    max_candidate_lines=5,
    max_storage_sites=2,
    keep_scenarios=None,
    keep_periods=None,
):
    """构造受限许可证可用的数据压缩函数。"""
    if keep_scenarios is None:
        keep_scenarios = ["S3", "S6"]
    if keep_periods is None:
        keep_periods = [1, 2, 3]

    def _transform(data):
        return shrink_tnep_data(
            data,
            max_candidate_lines=max_candidate_lines,
            max_storage_sites=max_storage_sites,
            keep_scenarios=keep_scenarios,
            keep_periods=keep_periods,
        )

    return _transform
