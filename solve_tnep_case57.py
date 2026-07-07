"""读取 dataset_csv_case57 并求解 TNEP + 储能联合规划 MILP。"""

import argparse
import os
import sys
import time
from collections import defaultdict
import math
import json

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Reals,
    Set,
    SolverFactory,
    Var,
    minimize,
    value,
)
import csv


DEFAULT_CSV_DIR = os.path.join(os.path.dirname(__file__), "dataset_csv_rts_gmlc")
DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "debug-aa4b49.log")


def _debug_log(run_id, hypothesis_id, location, message, data):
    payload = {
        "sessionId": "aa4b49",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def load_tnep_data_from_csv(csv_dir):
    params = {}
    with open(os.path.join(csv_dir, "global_params.csv"), "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            params[row["parameter"]] = float(row["value"])

    buses = []
    buses_kv = {}
    with open(os.path.join(csv_dir, "buses.csv"), "r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            b = int(r["bus"])
            buses.append(b)
            buses_kv[b] = float(r.get("base_kv", 230.0))

    existing_lines = []
    with open(os.path.join(csv_dir, "existing_lines.csv"), "r", newline="", encoding="utf-8-sig") as f:
        existing_lines = [(int(r["from_bus"]), int(r["to_bus"])) for r in csv.DictReader(f)]

    candidate_lines, line_meta, c_line = [], {}, {}
    with open(os.path.join(csv_dir, "candidate_lines.csv"), "r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            i, j = int(r["from_bus"]), int(r["to_bus"])
            key = (i, j)
            candidate_lines.append(key)
            line_meta[key] = {
                "kv": int(float(r["kv"])),
                "hops": int(float(r["hops"])),
                "length_mile": float(r["length_mile"]),
                "annual_cost_usd": float(r["c_line_usd_per_year"]),
            }
            c_line[key] = line_meta[key]["annual_cost_usd"]

    c_gen, gen_meta = {}, {}
    with open(os.path.join(csv_dir, "generators.csv"), "r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            g = int(r["g"])
            c_gen[g] = float(r["c_gen_usd_per_mwh"])
            gen_meta[g] = {"bus": int(r["bus"]), "pmax": float(r["pmax"]), "pmin": float(r["pmin"]), "c_gen": c_gen[g]}

    storage_sites, storage_meta = [], {}
    with open(os.path.join(csv_dir, "storage_sites.csv"), "r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            h = int(r["h"])
            storage_sites.append(h)
            storage_meta[h] = {"Ebar": float(r["Ebar_mwh"]), "c_fix": float(r["c_fix_usd_per_year"])}

    scenarios, omega = [], {}
    with open(os.path.join(csv_dir, "scenarios.csv"), "r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            s = r["scenario"]
            scenarios.append(s)
            omega[s] = float(r["omega_days"])

    with open(os.path.join(csv_dir, "periods.csv"), "r", newline="", encoding="utf-8-sig") as f:
        periods = [int(r["t"]) for r in csv.DictReader(f)]

    demand = {}
    with open(os.path.join(csv_dir, "demand.csv"), "r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            demand[(int(r["bus"]), r["scenario"], int(r["t"]))] = float(r["demand_mw"])

    renewable_meta = {}
    with open(os.path.join(csv_dir, "renewables.csv"), "r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rid = r.get("r", r.get("ren_id"))
            renewable_meta[rid] = {
                "bus": int(r["bus"]),
                "cap": float(r.get("capacity_mw", r.get("cap_mw"))),
                "type": r.get("type", "wind"),
            }

    wbar = {}
    with open(os.path.join(csv_dir, "renewable_availability.csv"), "r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rid = r.get("r", r.get("ren_id"))
            w = r.get("wbar_mw", r.get("avail_mw"))
            wbar[(rid, r["scenario"], int(r["t"]))] = float(w)

    return {
        "buses": buses,
        "buses_kv": buses_kv,
        "base_mva": params["base_mva"],
        "existing_lines": existing_lines,
        "candidate_lines": candidate_lines,
        "line_meta": line_meta,
        "c_line": c_line,
        "c_cap": params["c_cap"],
        "c_shed": params["c_shed"],
        "c_curt": params["c_curt"],
        "Gamma": params["Gamma"],
        "rho": params["rho"],
        "eta_c": params["eta_c"],
        "eta_d": params["eta_d"],
        "delta_t": params["delta_t"],
        "c_gen": c_gen,
        "gen_meta": gen_meta,
        "storage_sites": storage_sites,
        "storage_meta": storage_meta,
        "scenarios": scenarios,
        "omega": omega,
        "periods": periods,
        "demand": demand,
        "renewable_meta": renewable_meta,
        "wbar": wbar,
    }


def infer_line_electric_params(tnep_data):
    buses_kv = tnep_data["buses_kv"]
    candidate_lines = tnep_data["candidate_lines"]
    existing_lines = tnep_data["existing_lines"]

    def kv_level(i, j):
        if buses_kv[i] >= 400 or buses_kv[j] >= 400:
            return 500
        if buses_kv[i] >= 200 and buses_kv[j] >= 200:
            return 230
        return 138

    fbar = {}
    b = {}
    for l in existing_lines + candidate_lines:
        i, j = l
        kv = kv_level(i, j)
        if kv == 500:
            x = 0.035
            f = 900.0
        elif kv == 230:
            x = 0.06
            f = 500.0
        else:
            x = 0.12
            f = 200.0
        b[l] = 1.0 / x
        fbar[l] = f
    return fbar, b


def build_model(csv_dir):
    tnep_data = load_tnep_data_from_csv(csv_dir)
    op = {
        "G": sorted(tnep_data["gen_meta"].keys()),
        "g_bus": {g: tnep_data["gen_meta"][g]["bus"] for g in tnep_data["gen_meta"]},
        "R": sorted(tnep_data["renewable_meta"].keys()),
        "r_bus": {r: tnep_data["renewable_meta"][r]["bus"] for r in tnep_data["renewable_meta"]},
        "S": tnep_data["scenarios"],
        "T": tnep_data["periods"],
        "omega": tnep_data["omega"],
        "delta_t": tnep_data["delta_t"],
        "D": tnep_data["demand"],
        "Wbar": tnep_data["wbar"],
        "H": tnep_data["storage_sites"],
        "Ebar": {h: tnep_data["storage_meta"][h]["Ebar"] for h in tnep_data["storage_sites"]},
        "c_fix_h": {h: tnep_data["storage_meta"][h]["c_fix"] for h in tnep_data["storage_sites"]},
        "rho": tnep_data["rho"],
        "eta_c": tnep_data["eta_c"],
        "eta_d": tnep_data["eta_d"],
        "b_ref": min(tnep_data["buses"]),
        "theta_max": 1.0471975512,
    }
    fbar, b_param = infer_line_electric_params(tnep_data)

    m = ConcreteModel()
    m.B = Set(initialize=tnep_data["buses"], ordered=True)
    m.L0 = Set(initialize=tnep_data["existing_lines"], dimen=2)
    m.LC = Set(initialize=tnep_data["candidate_lines"], dimen=2)
    m.L = m.L0 | m.LC
    m.G = Set(initialize=op["G"])
    m.R = Set(initialize=op["R"])
    m.H = Set(initialize=op["H"])
    m.S = Set(initialize=op["S"])
    m.T = Set(initialize=op["T"], ordered=True)
    m.T0 = Set(initialize=[0] + op["T"], ordered=True)

    m.c_line = Param(m.LC, initialize=tnep_data["c_line"])
    m.c_fix = Param(m.H, initialize=op["c_fix_h"])
    m.c_cap = Param(initialize=tnep_data["c_cap"])
    m.c_shed = Param(initialize=tnep_data["c_shed"])
    m.c_curt = Param(initialize=tnep_data["c_curt"])
    m.Gamma = Param(initialize=tnep_data["Gamma"])
    m.delta_t = Param(initialize=op["delta_t"])
    m.theta_max = Param(initialize=op["theta_max"])

    m.Fbar = Param(m.L, initialize=fbar)
    m.Bline = Param(m.L, initialize={l: b_param[l] * tnep_data["base_mva"] for l in b_param})
    m.M = Param(m.LC, initialize={l: b_param[l] * op["theta_max"] * tnep_data["base_mva"] for l in tnep_data["candidate_lines"]})

    m.Pmax = Param(m.G, initialize={g: op_v["pmax"] for g, op_v in tnep_data["gen_meta"].items()})
    m.c_gen = Param(m.G, initialize=tnep_data["c_gen"])
    m.omega = Param(m.S, initialize=op["omega"])
    m.Ebar = Param(m.H, initialize=op["Ebar"])
    m.rho = Param(initialize=op["rho"])
    m.eta_c = Param(initialize=op["eta_c"])
    m.eta_d = Param(initialize=op["eta_d"])
    m.b_ref = Param(initialize=op["b_ref"])

    m.D = Param(m.B, m.S, m.T, initialize=op["D"])
    m.Wbar = Param(m.R, m.S, m.T, initialize=op["Wbar"])

    m.y = Var(m.LC, domain=Binary)
    m.z = Var(m.H, domain=Binary)
    m.E = Var(m.H, domain=NonNegativeReals)
    m.f = Var(m.L, m.S, m.T, domain=Reals)
    m.p = Var(m.G, m.S, m.T, domain=NonNegativeReals)
    m.w = Var(m.R, m.S, m.T, domain=NonNegativeReals)
    m.theta = Var(m.B, m.S, m.T, domain=Reals, bounds=(-3.2, 3.2))
    m.d_shed = Var(m.B, m.S, m.T, domain=NonNegativeReals)
    m.e_plus = Var(m.H, m.S, m.T, domain=NonNegativeReals)
    m.e_minus = Var(m.H, m.S, m.T, domain=NonNegativeReals)
    m.q = Var(m.H, m.S, m.T0, domain=NonNegativeReals)

    g_bus = op["g_bus"]
    r_bus = op["r_bus"]
    out_lines = {b: [l for l in m.L if l[0] == b] for b in m.B}
    in_lines = {b: [l for l in m.L if l[1] == b] for b in m.B}

    def obj_rule(model):
        invest = sum(model.c_line[l] * model.y[l] for l in model.LC) + sum(model.c_fix[h] * model.z[h] + model.c_cap * model.E[h] for h in model.H)
        op_cost = sum(
            model.omega[s]
            * sum(
                (
                    sum(model.c_gen[g] * model.p[g, s, t] * model.delta_t for g in model.G)
                    + model.c_shed * sum(model.d_shed[b, s, t] * model.delta_t for b in model.B)
                    + model.c_curt * sum((model.Wbar[r, s, t] - model.w[r, s, t]) * model.delta_t for r in model.R)
                )
                for t in model.T
            )
            for s in model.S
        )
        return invest + op_cost

    m.obj = Objective(rule=obj_rule, sense=minimize)

    def power_balance_rule(model, b, s, t):
        gen_term = sum(model.p[g, s, t] for g in model.G if g_bus[g] == b)
        ren_term = sum(model.w[r, s, t] for r in model.R if r_bus[r] == b)
        stor_term = sum(model.e_plus[h, s, t] - model.e_minus[h, s, t] for h in model.H if h == b)
        flow_term = sum(model.f[l, s, t] for l in out_lines[b]) - sum(model.f[l, s, t] for l in in_lines[b])
        return gen_term + ren_term + stor_term + model.d_shed[b, s, t] - model.D[b, s, t] == flow_term

    m.power_balance = Constraint(m.B, m.S, m.T, rule=power_balance_rule)
    m.dc_exist = Constraint(m.L0, m.S, m.T, rule=lambda model, i, j, s, t: model.f[(i, j), s, t] == model.Bline[(i, j)] * (model.theta[i, s, t] - model.theta[j, s, t]))
    m.dc_cand_up = Constraint(m.LC, m.S, m.T, rule=lambda model, i, j, s, t: model.f[(i, j), s, t] - model.Bline[(i, j)] * (model.theta[i, s, t] - model.theta[j, s, t]) <= model.M[(i, j)] * (1 - model.y[(i, j)]))
    m.dc_cand_dn = Constraint(m.LC, m.S, m.T, rule=lambda model, i, j, s, t: model.f[(i, j), s, t] - model.Bline[(i, j)] * (model.theta[i, s, t] - model.theta[j, s, t]) >= -model.M[(i, j)] * (1 - model.y[(i, j)]))
    m.thermal_exist_up = Constraint(m.L0, m.S, m.T, rule=lambda model, i, j, s, t: model.f[(i, j), s, t] <= model.Fbar[(i, j)])
    m.thermal_exist_dn = Constraint(m.L0, m.S, m.T, rule=lambda model, i, j, s, t: model.f[(i, j), s, t] >= -model.Fbar[(i, j)])
    m.thermal_cand_up = Constraint(m.LC, m.S, m.T, rule=lambda model, i, j, s, t: model.f[(i, j), s, t] <= model.Fbar[(i, j)] * model.y[(i, j)])
    m.thermal_cand_dn = Constraint(m.LC, m.S, m.T, rule=lambda model, i, j, s, t: model.f[(i, j), s, t] >= -model.Fbar[(i, j)] * model.y[(i, j)])
    m.gen_bound = Constraint(m.G, m.S, m.T, rule=lambda model, g, s, t: model.p[g, s, t] <= model.Pmax[g])
    m.ren_bound = Constraint(m.R, m.S, m.T, rule=lambda model, r, s, t: model.w[r, s, t] <= model.Wbar[r, s, t])
    m.stor_discharge = Constraint(m.H, m.S, m.T, rule=lambda model, h, s, t: model.e_plus[h, s, t] <= model.rho * model.E[h])
    m.stor_charge = Constraint(m.H, m.S, m.T, rule=lambda model, h, s, t: model.e_minus[h, s, t] <= model.rho * model.E[h])
    m.soc_dyn = Constraint(m.H, m.S, m.T, rule=lambda model, h, s, t: model.q[h, s, t] == model.q[h, s, t - 1] + model.eta_c * model.e_minus[h, s, t] * model.delta_t - (model.e_plus[h, s, t] / model.eta_d) * model.delta_t)
    m.soc_upper = Constraint(m.H, m.S, m.T0, rule=lambda model, h, s, t: model.q[h, s, t] <= model.E[h])
    m.soc_cyclic = Constraint(m.H, m.S, rule=lambda model, h, s: model.q[h, s, 0] == model.q[h, s, max(model.T)])
    m.stor_link = Constraint(m.H, rule=lambda model, h: model.E[h] <= model.Ebar[h] * model.z[h])
    m.budget = Constraint(rule=lambda model: sum(model.c_line[l] * model.y[l] for l in model.LC) + sum(model.c_fix[h] * model.z[h] + model.c_cap * model.E[h] for h in model.H) <= model.Gamma)
    m.shed_bound = Constraint(m.B, m.S, m.T, rule=lambda model, b, s, t: model.d_shed[b, s, t] <= model.D[b, s, t])
    m.ref_bus = Constraint(m.S, m.T, rule=lambda model, s, t: model.theta[model.b_ref, s, t] == 0.0)
    m.ang_diff_up = Constraint(m.L, m.S, m.T, rule=lambda model, i, j, s, t: model.theta[i, s, t] - model.theta[j, s, t] <= model.theta_max)
    m.ang_diff_dn = Constraint(m.L, m.S, m.T, rule=lambda model, i, j, s, t: model.theta[i, s, t] - model.theta[j, s, t] >= -model.theta_max)
    return m, tnep_data


def pick_solver(preferred="auto"):
    if preferred and preferred != "auto":
        if SolverFactory(preferred).available(False):
            return preferred
        raise RuntimeError(f"指定求解器不可用: {preferred}")

    for name in ("gurobi", "highs", "appsi_highs", "cbc", "glpk"):
        if SolverFactory(name).available(False):
            return name
    py = sys.executable
    raise RuntimeError(
        "未检测到可用MILP求解器。"
        f"\n当前Python解释器: {py}"
        "\n建议先执行："
        f'\n"{py}" -m pip install highspy'
    )


def _normalize_time_limit(time_limit):
    if time_limit is None:
        return None
    try:
        tl = float(time_limit)
    except Exception:
        return None
    if tl <= 0:
        return None
    return tl


def configure_solver(solver_name, time_limit=120, mip_rel_gap=0.01, gurobi_seed=None):
    solver = SolverFactory(solver_name)
    normalized_tl = _normalize_time_limit(time_limit)
    if hasattr(solver, "config"):
        try:
            solver.config.time_limit = normalized_tl
            solver.config.rel_gap = mip_rel_gap
            solver.config.raise_exception_on_nonoptimal_result = False
        except Exception:
            pass
    if solver_name in ("appsi_highs", "highs"):
        if solver_name == "appsi_highs":
            if normalized_tl is not None:
                solver.highs_options["time_limit"] = normalized_tl
            solver.highs_options["mip_rel_gap"] = mip_rel_gap
        else:
            if normalized_tl is not None:
                solver.options["time_limit"] = normalized_tl
            elif "time_limit" in solver.options:
                del solver.options["time_limit"]
            solver.options["mip_rel_gap"] = mip_rel_gap
    elif solver_name == "gurobi":
        if normalized_tl is not None:
            solver.options["TimeLimit"] = normalized_tl
        elif "TimeLimit" in solver.options:
            del solver.options["TimeLimit"]
        solver.options["MIPGap"] = mip_rel_gap
        if gurobi_seed is not None:
            solver.options["Seed"] = int(gurobi_seed)
    return solver


def _extract_bounds_and_gap(result):
    try:
        p = result["Problem"][0]
        lb = float(p["Lower bound"])
        ub = float(p["Upper bound"])
        if math.isfinite(lb) and math.isfinite(ub):
            denom = max(abs(ub), 1e-9)
            gap = abs(ub - lb) / denom
            return lb, ub, gap
        return lb, ub, None
    except Exception:
        return None, None, None


def _build_bus_energy_stats(tnep_data):
    load_bus = defaultdict(float)
    ren_bus = defaultdict(float)
    ren_bus_map = defaultdict(list)
    for rid, meta in tnep_data["renewable_meta"].items():
        ren_bus_map[meta["bus"]].append(rid)

    for s in tnep_data["scenarios"]:
        w = tnep_data["omega"][s]
        for t in tnep_data["periods"]:
            for b in tnep_data["buses"]:
                load_bus[b] += w * tnep_data["demand"][(b, s, t)] * tnep_data["delta_t"]
            for b, rlist in ren_bus_map.items():
                ren_bus[b] += w * sum(tnep_data["wbar"][(rid, s, t)] for rid in rlist) * tnep_data["delta_t"]
    return load_bus, ren_bus, ren_bus_map


def _fallback_feasible_solution_metrics(tnep_data):
    """构造一个始终可行的基线方案（无扩建、无储能、零潮流）并计算指标。"""
    buses = tnep_data["buses"]
    scenarios = tnep_data["scenarios"]
    periods = tnep_data["periods"]
    omega = tnep_data["omega"]
    delta_t = tnep_data["delta_t"]
    c_shed = tnep_data["c_shed"]
    c_curt = tnep_data["c_curt"]

    gen_at_bus = defaultdict(list)
    for g, meta in tnep_data["gen_meta"].items():
        gen_at_bus[meta["bus"]].append((g, meta["pmax"], tnep_data["c_gen"][g]))
    for b in gen_at_bus:
        gen_at_bus[b].sort(key=lambda x: x[2])  # 按边际成本从低到高

    ren_at_bus = defaultdict(list)
    for rid, meta in tnep_data["renewable_meta"].items():
        ren_at_bus[meta["bus"]].append(rid)

    op_cost = 0.0
    shed_total = 0.0
    curt_total = 0.0
    gen_cost_total = 0.0

    for s in scenarios:
        w_days = omega[s]
        for t in periods:
            for b in buses:
                demand_bt = tnep_data["demand"][(b, s, t)]
                rids = ren_at_bus[b]
                ren_avail_bt = sum(tnep_data["wbar"][(rid, s, t)] for rid in rids)

                # 本地可再生优先消纳（无网络交换）
                ren_used = min(demand_bt, ren_avail_bt)
                remain = demand_bt - ren_used
                curt_bt = ren_avail_bt - ren_used

                # 本地常规机组补足
                gen_used = 0.0
                gen_cost_bt = 0.0
                if remain > 0:
                    for _, pmax, c_gen in gen_at_bus.get(b, []):
                        if remain <= 0:
                            break
                        take = min(remain, pmax)
                        gen_used += take
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


def _build_storage_scores(tnep_data):
    storage_scores = []
    for h in tnep_data["storage_sites"]:
        net_series = []
        for s in tnep_data["scenarios"]:
            for t in tnep_data["periods"]:
                load_val = tnep_data["demand"][(h, s, t)]
                ren_val = sum(
                    tnep_data["wbar"][(rid, s, t)]
                    for rid, meta in tnep_data["renewable_meta"].items()
                    if meta["bus"] == h
                )
                net_series.append(load_val - ren_val)
        swing = max(net_series) - min(net_series) if net_series else 0.0
        ebar = tnep_data["storage_meta"][h]["Ebar"]
        c_fix = tnep_data["storage_meta"][h]["c_fix"]
        cap_cost = tnep_data["c_cap"] * ebar + c_fix
        score = swing / max(1.0, cap_cost)
        storage_scores.append((score, h))
    storage_scores.sort(key=lambda x: x[0], reverse=True)
    return storage_scores


def select_heuristic_candidates(tnep_data, max_lines=10, max_storage=4, method="score"):
    load_bus, ren_bus, _ = _build_bus_energy_stats(tnep_data)
    net = {b: load_bus[b] - ren_bus[b] for b in tnep_data["buses"]}
    deficit = {b: max(0.0, net[b]) for b in tnep_data["buses"]}
    surplus = {b: max(0.0, -net[b]) for b in tnep_data["buses"]}

    line_scores = []
    for l in tnep_data["candidate_lines"]:
        i, j = l
        transfer_potential = max(deficit[i] * surplus[j], deficit[j] * surplus[i]) ** 0.5
        imbalance_gap = abs(net[i] - net[j])
        c = max(1.0, tnep_data["c_line"][l])
        if method == "bridge":
            # 强化“缺口母线-富余母线”的桥接能力。
            bridge = deficit[i] * surplus[j] + deficit[j] * surplus[i]
            score = (0.7 * transfer_potential + 0.2 * imbalance_gap + 0.1 * bridge**0.5) / c
        else:
            score = (transfer_potential + 0.5 * imbalance_gap) / c
        line_scores.append((score, l))
    line_scores.sort(key=lambda x: x[0], reverse=True)

    storage_scores = _build_storage_scores(tnep_data)

    if method == "diversity":
        # 多样性启发式：优先覆盖更多不同母线，避免投资集中在局部区域。
        selected_lines = []
        bus_used_count = defaultdict(int)
        target = min(max_lines, len(line_scores))
        while len(selected_lines) < target:
            best = None
            best_key = None
            for base_score, l in line_scores:
                if l in selected_lines:
                    continue
                i, j = l
                novelty = 2.0 / (1.0 + bus_used_count[i] + bus_used_count[j])
                key = base_score * novelty
                if best is None or key > best_key:
                    best = l
                    best_key = key
            if best is None:
                break
            selected_lines.append(best)
            bus_used_count[best[0]] += 1
            bus_used_count[best[1]] += 1

        selected_storage = []
        for _, h in storage_scores:
            if len(selected_storage) >= min(max_storage, len(storage_scores)):
                break
            if bus_used_count[h] == 0 or len(selected_storage) < max(1, max_storage // 2):
                selected_storage.append(h)
    elif method == "budget_greedy":
        # 按“得分/成本”在预算内贪心选取，避免启发式集合过大造成MIP难以出首个可行解。
        budget_guard = 0.9 * float(tnep_data["Gamma"])
        budget_used = 0.0
        selected_lines = []
        selected_storage = []
        for _, l in line_scores:
            if len(selected_lines) >= min(max_lines, len(line_scores)):
                break
            c = float(tnep_data["c_line"][l])
            if budget_used + c <= budget_guard:
                selected_lines.append(l)
                budget_used += c
        for _, h in storage_scores:
            if len(selected_storage) >= min(max_storage, len(storage_scores)):
                break
            c = float(tnep_data["storage_meta"][h]["c_fix"])
            if budget_used + c <= budget_guard:
                selected_storage.append(h)
                budget_used += c
    else:
        selected_lines = [l for _, l in line_scores[: min(max_lines, len(line_scores))]]
        selected_storage = [h for _, h in storage_scores[: min(max_storage, len(storage_scores))]]

    return selected_lines, selected_storage


def _apply_heuristic_fixings(
    model,
    tnep_data,
    heuristic_lines=10,
    heuristic_storage=4,
    strong_heuristic=False,
    heuristic_method="score",
):
    selected_lines, selected_storage = select_heuristic_candidates(
        tnep_data,
        max_lines=heuristic_lines,
        max_storage=heuristic_storage,
        method=heuristic_method,
    )
    selected_line_set = set(selected_lines)
    selected_storage_set = set(selected_storage)
    forced_line_set = set()
    forced_storage_set = set()
    mandatory_invest = 0.0

    if strong_heuristic:
        # 强启发式下将投资变量完全固定，但需保证固定后的刚性投资不超过预算，
        # 否则模型会直接不可行。
        budget_guard = 0.95 * float(tnep_data["Gamma"])
        for l in selected_lines:
            c = float(tnep_data["c_line"][l])
            if mandatory_invest + c <= budget_guard:
                forced_line_set.add(l)
                mandatory_invest += c
        for h in selected_storage:
            c = float(tnep_data["storage_meta"][h]["c_fix"])
            if mandatory_invest + c <= budget_guard:
                forced_storage_set.add(h)
                mandatory_invest += c

    for l in model.LC:
        if strong_heuristic:
            if l in forced_line_set:
                model.y[l].fix(1)
            else:
                model.y[l].fix(0)
        elif l not in selected_line_set:
            model.y[l].fix(0)

    for h in model.H:
        if strong_heuristic:
            if h in forced_storage_set:
                model.z[h].fix(1)
            else:
                model.z[h].fix(0)
                model.E[h].fix(0)
        elif h not in selected_storage_set:
            model.z[h].fix(0)
            model.E[h].fix(0)

    return {
        "selected_lines": selected_lines,
        "selected_storage": selected_storage,
        "forced_lines": sorted(forced_line_set),
        "forced_storage": sorted(forced_storage_set),
        "mandatory_invest": mandatory_invest,
    }


def solve_once(
    csv_dir,
    time_limit=120,
    heuristic=False,
    heuristic_lines=10,
    heuristic_storage=4,
    strong_heuristic=False,
    heuristic_method="score",
    solver_preference="auto",
    gurobi_seed=None,
):
    result = solve_and_collect(
        csv_dir,
        time_limit=time_limit,
        heuristic=heuristic,
        heuristic_lines=heuristic_lines,
        heuristic_storage=heuristic_storage,
        strong_heuristic=strong_heuristic,
        heuristic_method=heuristic_method,
        solver_preference=solver_preference,
        gurobi_seed=gurobi_seed,
        print_header=True,
    )
    _print_result_block("单次求解", result)


def solve_and_collect(
    csv_dir,
    time_limit=120,
    heuristic=False,
    heuristic_lines=10,
    heuristic_storage=4,
    strong_heuristic=False,
    heuristic_method="score",
    solver_preference="auto",
    gurobi_seed=None,
    print_header=True,
):
    model, tnep_data = build_model(csv_dir)
    preferred_solver = pick_solver(preferred=solver_preference)
    solver_candidates = [preferred_solver]
    if preferred_solver == "gurobi":
        for alt in ("highs", "appsi_highs"):
            if alt not in solver_candidates and SolverFactory(alt).available(False):
                solver_candidates.append(alt)
    run_id = "heuristic" if heuristic else "exact"
    if print_header:
        print(f"数据目录: {csv_dir}")
        print(f"首选求解器: {preferred_solver}")

    if heuristic:
        heuristic_fix_info = _apply_heuristic_fixings(
            model,
            tnep_data,
            heuristic_lines=heuristic_lines,
            heuristic_storage=heuristic_storage,
            strong_heuristic=strong_heuristic,
            heuristic_method=heuristic_method,
        )
        selected_lines = heuristic_fix_info["selected_lines"]
        selected_storage = heuristic_fix_info["selected_storage"]
        if print_header:
            print(
                f"启发式已启用: 线路候选 {len(selected_lines)}/{len(list(model.LC))}, "
                f"储能候选 {len(selected_storage)}/{len(list(model.H))}"
            )
            print(f"启发式策略: {heuristic_method}")
            if strong_heuristic:
                print(
                    "强启发式已启用: 固定线路 "
                    f"{len(heuristic_fix_info['forced_lines'])}/{len(selected_lines)}, "
                    f"固定储能 {len(heuristic_fix_info['forced_storage'])}/{len(selected_storage)}, "
                    f"刚性投资={heuristic_fix_info['mandatory_invest']:,.2f} USD/year"
                )

    result = None
    solve_time_sec = None
    solver_name = None
    last_exc = None
    for idx, cand_solver in enumerate(solver_candidates):
        solver_name = cand_solver
        solver = configure_solver(
            solver_name,
            time_limit=time_limit,
            mip_rel_gap=0.01,
            gurobi_seed=gurobi_seed,
        )
        if print_header:
            print(f"使用求解器: {solver_name}")
        # region agent log
        _debug_log(
            run_id,
            "H2",
            "solve_tnep_case57.py:solve_and_collect:before_solve",
            "solver configured",
            {
                "solver_name": solver_name,
                "time_limit_arg_sec": time_limit,
                "candidate_line_count": len(list(model.LC)),
                "storage_site_count": len(list(model.H)),
                "gurobi_seed": gurobi_seed,
            },
        )
        # endregion
        # region agent log
        _debug_log(
            run_id,
            "H5",
            "solve_tnep_case57.py:solve_and_collect:solver_effective_settings",
            "effective solver settings before solve",
            {
                "solver_name": solver_name,
                "config_time_limit": getattr(getattr(solver, "config", None), "time_limit", None),
                "config_rel_gap": getattr(getattr(solver, "config", None), "rel_gap", None),
                "config_raise_exception_on_nonoptimal_result": getattr(
                    getattr(solver, "config", None), "raise_exception_on_nonoptimal_result", None
                ),
                "options_time_limit": getattr(getattr(solver, "options", {}), "get", lambda _k, _d=None: None)("time_limit", None),
                "options_mip_rel_gap": getattr(getattr(solver, "options", {}), "get", lambda _k, _d=None: None)("mip_rel_gap", None),
                "options_gurobi_seed": getattr(getattr(solver, "options", {}), "get", lambda _k, _d=None: None)("Seed", None),
            },
        )
        # endregion
        t0 = time.perf_counter()
        try:
            result = solver.solve(model, load_solutions=False)
            solve_time_sec = time.perf_counter() - t0
            break
        except Exception as e:
            solve_time_sec = time.perf_counter() - t0
            last_exc = e
            # region agent log
            _debug_log(
                run_id,
                "H1",
                "solve_tnep_case57.py:solve_and_collect:solve_exception",
                "solver raised exception",
                {
                    "solver_name": solver_name,
                    "time_limit_arg_sec": time_limit,
                "gurobi_seed": gurobi_seed,
                    "elapsed_sec": solve_time_sec,
                    "error_type": type(e).__name__,
                    "error_text": str(e),
                },
            )
            # endregion
            if idx < len(solver_candidates) - 1 and print_header:
                print(f"求解器 {solver_name} 失败，尝试回退到 {solver_candidates[idx + 1]}。")
            result = None

    if result is None:
        e = last_exc if last_exc is not None else RuntimeError("unknown solver failure")
        return {
            "termination_condition": f"FAILED: {type(e).__name__}: {e}",
            "solve_time_sec": solve_time_sec if solve_time_sec is not None else 0.0,
            "lb": None,
            "ub": None,
            "mip_gap": None,
            "obj": None,
            "selected_lines": None,
            "selected_storage": None,
            "total_storage": None,
            "invest_line": None,
            "invest_stor": None,
            "shed_total": None,
            "curt_total": None,
            "gamma": None,
            "tnep_data": tnep_data,
            "ok": False,
            "solution_source": "solver_exception",
        }

    lb, ub, mip_gap = _extract_bounds_and_gap(result)
    solution_count = 0
    try:
        solution_count = len(result.solution)
    except Exception:
        solution_count = 0
    # region agent log
    _debug_log(
        run_id,
        "H6",
        "solve_tnep_case57.py:solve_and_collect:after_solve",
        "solver returned result object",
        {
            "solver_name": solver_name,
            "elapsed_sec": solve_time_sec,
            "solver_status": str(result.solver.status),
            "termination_condition": str(result.solver.termination_condition),
            "solution_count": solution_count,
            "lb": lb,
            "ub": ub,
            "mip_gap": mip_gap,
        },
    )
    # endregion

    if solution_count <= 0:
        fallback = _fallback_feasible_solution_metrics(tnep_data)
        # region agent log
        _debug_log(
            run_id,
            "H6",
            "solve_tnep_case57.py:solve_and_collect:no_incumbent",
            "no incumbent solution to load",
            {
                "solver_status": str(result.solver.status),
                "termination_condition": str(result.solver.termination_condition),
                "elapsed_sec": solve_time_sec,
            },
        )
        # endregion
        return {
            "termination_condition": f"{result.solver.termination_condition} (no incumbent, fallback feasible dispatch)",
            "solve_time_sec": solve_time_sec,
            "lb": lb,
            "ub": ub,
            "mip_gap": mip_gap,
            "obj": fallback["obj"],
            "selected_lines": fallback["selected_lines"],
            "selected_storage": fallback["selected_storage"],
            "total_storage": fallback["total_storage"],
            "invest_line": fallback["invest_line"],
            "invest_stor": fallback["invest_stor"],
            "shed_total": fallback["shed_total"],
            "curt_total": fallback["curt_total"],
            "gamma": value(model.Gamma),
            "tnep_data": tnep_data,
            "ok": True,
            "solution_source": "fallback_feasible_dispatch",
        }

    try:
        model.solutions.load_from(result)
    except Exception as e:
        # region agent log
        _debug_log(
            run_id,
            "H6",
            "solve_tnep_case57.py:solve_and_collect:load_solution_exception",
            "failed to load incumbent solution",
            {
                "error_type": type(e).__name__,
                "error_text": str(e),
                "solution_count": solution_count,
            },
        )
        # endregion
        return {
            "termination_condition": f"{result.solver.termination_condition} (solution load failed)",
            "solve_time_sec": solve_time_sec,
            "lb": lb,
            "ub": ub,
            "mip_gap": mip_gap,
            "obj": None,
            "selected_lines": None,
            "selected_storage": None,
            "total_storage": None,
            "invest_line": None,
            "invest_stor": None,
            "shed_total": None,
            "curt_total": None,
            "gamma": value(model.Gamma),
            "tnep_data": tnep_data,
            "ok": False,
            "solution_source": "solution_load_failed",
        }

    selected_lines = [l for l in model.LC if value(model.y[l]) > 0.5]
    selected_storage = [h for h in model.H if value(model.z[h]) > 0.5]
    total_storage = sum(value(model.E[h]) for h in model.H)
    invest_line = sum(value(model.c_line[l]) * value(model.y[l]) for l in model.LC)
    invest_stor = sum(value(model.c_fix[h]) * value(model.z[h]) + value(model.c_cap) * value(model.E[h]) for h in model.H)
    shed_total = sum(value(model.omega[s]) * sum(value(model.d_shed[b, s, t]) * value(model.delta_t) for b in model.B for t in model.T) for s in model.S)
    curt_total = sum(value(model.omega[s]) * sum((value(model.Wbar[r, s, t]) - value(model.w[r, s, t])) * value(model.delta_t) for r in model.R for t in model.T) for s in model.S)
    obj_value = value(model.obj)
    term_cond = str(result.solver.termination_condition)
    # region agent log
    _debug_log(
        run_id,
        "H4",
        "solve_tnep_case57.py:solve_and_collect:solution_summary",
        "computed solution metrics",
        {
            "objective": obj_value,
            "selected_line_count": len(selected_lines),
            "selected_storage_count": len(selected_storage),
            "eens": shed_total,
            "curtailment": curt_total,
        },
    )
    # endregion

    return {
        "termination_condition": term_cond,
        "solve_time_sec": solve_time_sec,
        "lb": lb,
        "ub": ub,
        "mip_gap": mip_gap,
        "obj": obj_value,
        "selected_lines": selected_lines,
        "selected_storage": selected_storage,
        "total_storage": total_storage,
        "invest_line": invest_line,
        "invest_stor": invest_stor,
        "shed_total": shed_total,
        "curt_total": curt_total,
        "gamma": value(model.Gamma),
        "tnep_data": tnep_data,
        "ok": True,
        "solution_source": "incumbent",
    }


def _print_result_block(name, result):
    print(f"\n=== {name}：求解状态 ===")
    print(result["termination_condition"])
    print(f"结果来源: {result.get('solution_source', 'unknown')}")
    print(f"求解时间: {result['solve_time_sec']:.2f} 秒")
    if result["lb"] is not None and result["ub"] is not None:
        print(f"下界(LB): {result['lb']:,.2f}")
        print(f"上界(UB): {result['ub']:,.2f}")
    if result["mip_gap"] is not None:
        print(f"MIP GAP: {result['mip_gap']:.4%}")
    else:
        print("MIP GAP: N/A")
    if result["obj"] is None:
        print("目标函数值: N/A")
    else:
        print(f"目标函数值: {result['obj']:,.2f}")
    if result["selected_lines"] is None:
        print("已选候选线路数: N/A")
    else:
        print(f"已选候选线路数: {len(result['selected_lines'])}")
    if result["selected_storage"] is None:
        print("已选储能站点数: N/A")
    else:
        print(f"已选储能站点数: {len(result['selected_storage'])}")
    if result["total_storage"] is None:
        print("储能装机容量: N/A")
    else:
        print(f"储能装机容量: {result['total_storage']:.2f} MWh")
    if result["invest_line"] is None:
        print("线路年化投资: N/A")
    else:
        print(f"线路年化投资: {result['invest_line']:,.2f} USD/year")
    if result["invest_stor"] is None:
        print("储能年化投资: N/A")
    else:
        print(f"储能年化投资: {result['invest_stor']:,.2f} USD/year")
    if result["gamma"] is None:
        print("预算上限: N/A")
    else:
        print(f"预算上限: {result['gamma']:,.2f} USD/year")
    if result["shed_total"] is None:
        print("年期望失供电量 (EENS): N/A")
    else:
        print(f"年期望失供电量 (EENS): {result['shed_total']:,.2f} MWh/year")
    if result["curt_total"] is None:
        print("年期望新能源弃电量: N/A")
    else:
        print(f"年期望新能源弃电量: {result['curt_total']:,.2f} MWh/year")


def solve_compare(
    csv_dir,
    time_limit=120,
    heuristic_lines=10,
    heuristic_storage=4,
    exact_time_limit=None,
    heuristic_time_limit=None,
    strong_heuristic=False,
    heuristic_method="score",
    solver_preference="auto",
    gurobi_seed=None,
):
    exact_limit = time_limit if exact_time_limit is None else exact_time_limit
    heur_limit = time_limit if heuristic_time_limit is None else heuristic_time_limit
    exact_tl_norm = _normalize_time_limit(exact_limit)
    heur_tl_norm = _normalize_time_limit(heur_limit)

    print(f"数据目录: {csv_dir}")
    print("开始对比：精确求解 vs 启发式求解")
    exact_tl_text = "无上限" if exact_tl_norm is None else f"{int(exact_tl_norm)}s"
    heur_tl_text = "无上限" if heur_tl_norm is None else f"{int(heur_tl_norm)}s"
    print(f"时间上限: 精确={exact_tl_text}, 启发式={heur_tl_text}")
    print(f"启发式策略: {heuristic_method}")

    exact = solve_and_collect(
        csv_dir,
        time_limit=exact_tl_norm,
        heuristic=False,
        heuristic_lines=heuristic_lines,
        heuristic_storage=heuristic_storage,
        solver_preference=solver_preference,
        gurobi_seed=gurobi_seed,
        print_header=False,
    )
    heur = solve_and_collect(
        csv_dir,
        time_limit=heur_tl_norm,
        heuristic=True,
        heuristic_lines=heuristic_lines,
        heuristic_storage=heuristic_storage,
        strong_heuristic=strong_heuristic,
        heuristic_method=heuristic_method,
        solver_preference=solver_preference,
        gurobi_seed=gurobi_seed,
        print_header=False,
    )

    _print_result_block("精确求解", exact)
    _print_result_block("启发式求解", heur)

    print("\n=== 对比结论 ===")
    time_speedup = exact["solve_time_sec"] / max(heur["solve_time_sec"], 1e-9)
    if exact["obj"] is not None and heur["obj"] is not None:
        obj_diff = heur["obj"] - exact["obj"]
        rel_obj_diff = obj_diff / max(abs(exact["obj"]), 1e-9)
        print(f"目标值差异(启发式-精确): {obj_diff:,.2f} ({rel_obj_diff:.4%})")
    else:
        print("目标值差异(启发式-精确): N/A（至少一侧无可行解）")
    print(f"耗时比(精确/启发式): {time_speedup:.2f}x")
    exact_line_cnt = "N/A" if exact["selected_lines"] is None else str(len(exact["selected_lines"]))
    heur_line_cnt = "N/A" if heur["selected_lines"] is None else str(len(heur["selected_lines"]))
    exact_stor_cnt = "N/A" if exact["selected_storage"] is None else str(len(exact["selected_storage"]))
    heur_stor_cnt = "N/A" if heur["selected_storage"] is None else str(len(heur["selected_storage"]))
    print(f"线路数量对比: 精确={exact_line_cnt}, 启发式={heur_line_cnt}")
    print(f"储能站点对比: 精确={exact_stor_cnt}, 启发式={heur_stor_cnt}")
    if exact["shed_total"] is None or heur["shed_total"] is None:
        print("EENS 对比 (MWh/year): N/A（至少一侧无可行解）")
    else:
        print(
            f"EENS 对比 (MWh/year): 精确={exact['shed_total']:,.2f}, "
            f"启发式={heur['shed_total']:,.2f}"
        )
    if exact["curt_total"] is None or heur["curt_total"] is None:
        print("弃电对比 (MWh/year): N/A（至少一侧无可行解）")
    else:
        print(
            f"弃电对比 (MWh/year): 精确={exact['curt_total']:,.2f}, "
            f"启发式={heur['curt_total']:,.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default=DEFAULT_CSV_DIR, help="数据目录，默认 dataset_csv_case57")
    parser.add_argument("--time-limit", type=int, default=120, help="求解时间上限（秒）；设为0表示不设上限")
    parser.add_argument("--heuristic", action="store_true", help="启发式筛选投资候选后再求解")
    parser.add_argument("--compare", action="store_true", help="同时运行精确求解与启发式求解并输出对比")
    parser.add_argument("--strong-heuristic", action="store_true", help="强启发式：固定启发式选中投资决策，优先快速得到可行解")
    parser.add_argument("--exact-time-limit", type=int, default=None, help="compare模式下精确求解时间上限（秒）；设为0表示不设上限，默认继承 --time-limit")
    parser.add_argument("--heuristic-time-limit", type=int, default=None, help="compare模式下启发式求解时间上限（秒）；设为0表示不设上限，默认继承 --time-limit")
    parser.add_argument("--heuristic-lines", type=int, default=10, help="启发式保留的候选线路数量")
    parser.add_argument("--heuristic-storage", type=int, default=4, help="启发式保留的储能候选数量")
    parser.add_argument("--heuristic-method", choices=["score", "bridge", "budget_greedy", "diversity"], default="score", help="启发式策略：score(原始打分), bridge(供需桥接), budget_greedy(预算贪心), diversity(多样性覆盖)")
    parser.add_argument("--solver", choices=["auto", "highs", "appsi_highs", "gurobi"], default="auto", help="指定求解器；auto为自动选择并在gurobi失败时回退")
    parser.add_argument("--gurobi-seed", type=int, default=None, help="Gurobi随机种子（用于重复实验稳定性评估）")
    args = parser.parse_args()
    if args.compare:
        solve_compare(
            args.csv_dir,
            time_limit=args.time_limit,
            heuristic_lines=args.heuristic_lines,
            heuristic_storage=args.heuristic_storage,
            exact_time_limit=args.exact_time_limit,
            heuristic_time_limit=args.heuristic_time_limit,
            strong_heuristic=args.strong_heuristic,
            heuristic_method=args.heuristic_method,
            solver_preference=args.solver,
            gurobi_seed=args.gurobi_seed,
        )
    else:
        solve_once(
            args.csv_dir,
            time_limit=args.time_limit,
            heuristic=args.heuristic,
            heuristic_lines=args.heuristic_lines,
            heuristic_storage=args.heuristic_storage,
            strong_heuristic=args.strong_heuristic,
            heuristic_method=args.heuristic_method,
            solver_preference=args.solver,
            gurobi_seed=args.gurobi_seed,
        )
