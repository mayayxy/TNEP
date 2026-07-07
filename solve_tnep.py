"""按论文模型构建并求解 TNEP + 储能联合规划 MILP。

数据来源：
    dataset_csv/*.csv（由 data.py 生成）
运行方式：
    python solve_tnep.py
"""
import sys

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

from data import load_tnep_data_from_csv, CSV_DIR


def infer_line_electric_params(tnep_data):
    """为线路补充 Fbar 与 B（来自电压等级近似）。"""
    buses = tnep_data["buses"]
    candidate_lines = tnep_data["candidate_lines"]
    existing_lines = tnep_data["existing_lines"]

    # 与 data.py 中一致：1-10 为138kV，11-24为230kV
    bus_kv = {b: (138.0 if b <= 10 else 230.0) for b in buses}

    def kv_level(i, j):
        if bus_kv[i] >= 200 and bus_kv[j] >= 200:
            return 230
        if bus_kv[i] < 200 and bus_kv[j] < 200:
            return 138
        return 161  # 近似表示联络变压等级

    fbar = {}
    b = {}
    for l in existing_lines + candidate_lines:
        i, j = l
        kv = kv_level(i, j)
        if kv == 230:
            x = 0.05
            f = 500.0
        elif kv == 138:
            x = 0.12
            f = 175.0
        else:  # 联络通道
            x = 0.08
            f = 400.0
        b[l] = 1.0 / x
        fbar[l] = f
    return fbar, b


def build_model():
    tnep_data = load_tnep_data_from_csv(CSV_DIR)
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
        "b_ref": 13,
        "theta_max": 1.0471975512,
    }
    fbar, b_param = infer_line_electric_params(tnep_data)

    m = ConcreteModel()

    # ---------------- Sets ----------------
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

    # ---------------- Params ----------------
    m.c_line = Param(m.LC, initialize=tnep_data["c_line"])
    m.c_fix = Param(m.H, initialize=op["c_fix_h"])
    m.c_cap = Param(initialize=tnep_data["c_cap"])
    m.c_shed = Param(initialize=tnep_data["c_shed"])
    m.c_curt = Param(initialize=tnep_data["c_curt"])
    m.Gamma = Param(initialize=tnep_data["Gamma"])
    m.delta_t = Param(initialize=op["delta_t"])
    m.theta_max = Param(initialize=op["theta_max"])

    m.Fbar = Param(m.L, initialize=fbar)
    m.Bline = Param(
        m.L,
        initialize={l: b_param[l] * tnep_data["base_mva"] for l in b_param},
    )
    m.M = Param(
        m.LC,
        initialize={l: b_param[l] * op["theta_max"] * tnep_data["base_mva"] for l in tnep_data["candidate_lines"]},
    )

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

    # ---------------- Vars ----------------
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

    # ---------------- Helper maps ----------------
    g_bus = op["g_bus"]
    r_bus = op["r_bus"]

    out_lines = {b: [l for l in m.L if l[0] == b] for b in m.B}
    in_lines = {b: [l for l in m.L if l[1] == b] for b in m.B}

    # ---------------- Objective ----------------
    def obj_rule(model):
        invest = sum(model.c_line[l] * model.y[l] for l in model.LC) + sum(
            model.c_fix[h] * model.z[h] + model.c_cap * model.E[h] for h in model.H
        )
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

    # ---------------- Constraints ----------------
    def power_balance_rule(model, b, s, t):
        gen_term = sum(model.p[g, s, t] for g in model.G if g_bus[g] == b)
        ren_term = sum(model.w[r, s, t] for r in model.R if r_bus[r] == b)
        stor_term = sum(
            model.e_plus[h, s, t] - model.e_minus[h, s, t] for h in model.H if h == b
        )
        flow_term = sum(model.f[l, s, t] for l in out_lines[b]) - sum(model.f[l, s, t] for l in in_lines[b])
        return gen_term + ren_term + stor_term + model.d_shed[b, s, t] - model.D[b, s, t] == flow_term

    m.power_balance = Constraint(m.B, m.S, m.T, rule=power_balance_rule)

    def dc_exist_rule(model, i, j, s, t):
        return model.f[(i, j), s, t] == model.Bline[(i, j)] * (model.theta[i, s, t] - model.theta[j, s, t])

    m.dc_exist = Constraint(m.L0, m.S, m.T, rule=dc_exist_rule)

    def dc_cand_up_rule(model, i, j, s, t):
        return (
            model.f[(i, j), s, t] - model.Bline[(i, j)] * (model.theta[i, s, t] - model.theta[j, s, t])
            <= model.M[(i, j)] * (1 - model.y[(i, j)])
        )

    def dc_cand_dn_rule(model, i, j, s, t):
        return (
            model.f[(i, j), s, t] - model.Bline[(i, j)] * (model.theta[i, s, t] - model.theta[j, s, t])
            >= -model.M[(i, j)] * (1 - model.y[(i, j)])
        )

    m.dc_cand_up = Constraint(m.LC, m.S, m.T, rule=dc_cand_up_rule)
    m.dc_cand_dn = Constraint(m.LC, m.S, m.T, rule=dc_cand_dn_rule)

    def thermal_exist_up(model, i, j, s, t):
        return model.f[(i, j), s, t] <= model.Fbar[(i, j)]

    def thermal_exist_dn(model, i, j, s, t):
        return model.f[(i, j), s, t] >= -model.Fbar[(i, j)]

    m.thermal_exist_up = Constraint(m.L0, m.S, m.T, rule=thermal_exist_up)
    m.thermal_exist_dn = Constraint(m.L0, m.S, m.T, rule=thermal_exist_dn)

    def thermal_cand_up(model, i, j, s, t):
        return model.f[(i, j), s, t] <= model.Fbar[(i, j)] * model.y[(i, j)]

    def thermal_cand_dn(model, i, j, s, t):
        return model.f[(i, j), s, t] >= -model.Fbar[(i, j)] * model.y[(i, j)]

    m.thermal_cand_up = Constraint(m.LC, m.S, m.T, rule=thermal_cand_up)
    m.thermal_cand_dn = Constraint(m.LC, m.S, m.T, rule=thermal_cand_dn)

    def gen_bound_rule(model, g, s, t):
        return model.p[g, s, t] <= model.Pmax[g]

    m.gen_bound = Constraint(m.G, m.S, m.T, rule=gen_bound_rule)

    def ren_bound_rule(model, r, s, t):
        return model.w[r, s, t] <= model.Wbar[r, s, t]

    m.ren_bound = Constraint(m.R, m.S, m.T, rule=ren_bound_rule)

    def stor_discharge_rule(model, h, s, t):
        return model.e_plus[h, s, t] <= model.rho * model.E[h]

    def stor_charge_rule(model, h, s, t):
        return model.e_minus[h, s, t] <= model.rho * model.E[h]

    m.stor_discharge = Constraint(m.H, m.S, m.T, rule=stor_discharge_rule)
    m.stor_charge = Constraint(m.H, m.S, m.T, rule=stor_charge_rule)

    def soc_dyn_rule(model, h, s, t):
        return model.q[h, s, t] == model.q[h, s, t - 1] + model.eta_c * model.e_minus[h, s, t] * model.delta_t - (
            model.e_plus[h, s, t] / model.eta_d
        ) * model.delta_t

    m.soc_dyn = Constraint(m.H, m.S, m.T, rule=soc_dyn_rule)

    def soc_upper_rule(model, h, s, t):
        return model.q[h, s, t] <= model.E[h]

    m.soc_upper = Constraint(m.H, m.S, m.T0, rule=soc_upper_rule)

    def soc_cyclic_rule(model, h, s):
        t_last = max(model.T)
        return model.q[h, s, 0] == model.q[h, s, t_last]

    m.soc_cyclic = Constraint(m.H, m.S, rule=soc_cyclic_rule)

    def stor_link_rule(model, h):
        return model.E[h] <= model.Ebar[h] * model.z[h]

    m.stor_link = Constraint(m.H, rule=stor_link_rule)

    def budget_rule(model):
        return (
            sum(model.c_line[l] * model.y[l] for l in model.LC)
            + sum(model.c_fix[h] * model.z[h] + model.c_cap * model.E[h] for h in model.H)
            <= model.Gamma
        )

    m.budget = Constraint(rule=budget_rule)

    def shed_bound_rule(model, b, s, t):
        return model.d_shed[b, s, t] <= model.D[b, s, t]

    m.shed_bound = Constraint(m.B, m.S, m.T, rule=shed_bound_rule)

    def ref_bus_rule(model, s, t):
        return model.theta[model.b_ref, s, t] == 0.0

    m.ref_bus = Constraint(m.S, m.T, rule=ref_bus_rule)

    # 角差约束：用于限制DC近似与Big-M范围
    def ang_diff_up_rule(model, i, j, s, t):
        return model.theta[i, s, t] - model.theta[j, s, t] <= model.theta_max

    def ang_diff_dn_rule(model, i, j, s, t):
        return model.theta[i, s, t] - model.theta[j, s, t] >= -model.theta_max

    m.ang_diff_up = Constraint(m.L, m.S, m.T, rule=ang_diff_up_rule)
    m.ang_diff_dn = Constraint(m.L, m.S, m.T, rule=ang_diff_dn_rule)

    return m, tnep_data


def _pick_solver():
    solver_name = None
    for name in ("appsi_highs", "highs", "gurobi", "cbc", "glpk"):
        if SolverFactory(name).available(False):
            solver_name = name
            break

    if solver_name is None:
        py = sys.executable
        raise RuntimeError(
            "未检测到可用MILP求解器。"
            f"\n当前Python解释器: {py}"
            "\n建议先执行："
            f'\n"{py}" -m pip install highspy'
            "\n然后重新运行 solve_tnep.py。"
        )
    return solver_name


def _configure_solver(solver_name, time_limit=90, mip_rel_gap=0.005):
    solver = SolverFactory(solver_name)
    if solver_name in ("appsi_highs", "highs"):
        if solver_name == "appsi_highs":
            solver.highs_options["time_limit"] = time_limit
            solver.highs_options["mip_rel_gap"] = mip_rel_gap
        else:
            solver.options["time_limit"] = time_limit
            solver.options["mip_rel_gap"] = mip_rel_gap
    return solver


def _apply_case_investment_policy(model, allow_line, allow_storage):
    if not allow_line:
        for l in model.LC:
            model.y[l].fix(0)
    if not allow_storage:
        for h in model.H:
            model.z[h].fix(0)
            model.E[h].fix(0)


def _collect_metrics(model):
    selected_lines = [l for l in model.LC if value(model.y[l]) > 0.5]
    selected_storage = [h for h in model.H if value(model.z[h]) > 0.5]
    total_storage = sum(value(model.E[h]) for h in model.H)

    invest_line = sum(value(model.c_line[l]) * value(model.y[l]) for l in model.LC)
    invest_stor = sum(value(model.c_fix[h]) * value(model.z[h]) + value(model.c_cap) * value(model.E[h]) for h in model.H)
    shed_total = sum(
        value(model.omega[s]) * sum(value(model.d_shed[b, s, t]) * value(model.delta_t) for b in model.B for t in model.T)
        for s in model.S
    )
    curt_total = sum(
        value(model.omega[s]) * sum(
            (value(model.Wbar[r, s, t]) - value(model.w[r, s, t])) * value(model.delta_t)
            for r in model.R for t in model.T
        )
        for s in model.S
    )

    total_obj = value(model.obj)
    op_cost = total_obj - invest_line - invest_stor
    return {
        "objective": total_obj,
        "op_cost": op_cost,
        "invest_line": invest_line,
        "invest_storage": invest_stor,
        "selected_lines": selected_lines,
        "selected_storage": selected_storage,
        "total_storage": total_storage,
        "eens_mwh": shed_total,
        "curtailment_mwh": curt_total,
    }


def _solve_single_case(case_name, allow_line, allow_storage, solver_name, time_limit=90, mip_rel_gap=0.005, verbose=True):
    model, tnep_data = build_model()
    _apply_case_investment_policy(model, allow_line=allow_line, allow_storage=allow_storage)
    solver = _configure_solver(solver_name, time_limit=time_limit, mip_rel_gap=mip_rel_gap)
    result = solver.solve(model)
    metrics = _collect_metrics(model)
    term = str(result.solver.termination_condition)

    if verbose:
        print(f"\n=== 案例: {case_name} ===")
        print(f"终止状态: {term}")
        print(f"目标函数值: {metrics['objective']:,.2f}")
        print(f"线路年化投资: {metrics['invest_line']:,.2f} USD/year")
        print(f"储能年化投资: {metrics['invest_storage']:,.2f} USD/year")
        print(f"运行成本(估算): {metrics['op_cost']:,.2f} USD/year")
        print(f"EENS: {metrics['eens_mwh']:,.2f} MWh/year")
        print(f"弃电量: {metrics['curtailment_mwh']:,.2f} MWh/year")
        print(f"已选线路数: {len(metrics['selected_lines'])}")
        print(f"已选储能站点数: {len(metrics['selected_storage'])}")
        if metrics["selected_lines"]:
            print("线路样例(前10):", metrics["selected_lines"][:10])
        if metrics["selected_storage"]:
            print("储能站点:", metrics["selected_storage"])

    return {
        "case": case_name,
        "termination": term,
        "metrics": metrics,
        "tnep_data": tnep_data,
    }


def run_comparison_experiments():
    solver_name = _pick_solver()
    print(f"使用求解器: {solver_name}")
    print("\n开始对比实验: No-Invest / Only-Line / Only-Storage / Joint")

    cases = [
        ("No-Invest", False, False),
        ("Only-Line", True, False),
        ("Only-Storage", False, True),
        ("Joint", True, True),
    ]
    results = []
    for case_name, allow_line, allow_storage in cases:
        case_result = _solve_single_case(
            case_name=case_name,
            allow_line=allow_line,
            allow_storage=allow_storage,
            solver_name=solver_name,
            time_limit=90,
            mip_rel_gap=0.005,
            verbose=True,
        )
        results.append(case_result)

    baseline = next(r for r in results if r["case"] == "No-Invest")
    base_obj = baseline["metrics"]["objective"]
    base_eens = baseline["metrics"]["eens_mwh"]

    print("\n=== 对比汇总 (相对 No-Invest) ===")
    print("Case | Termination | Obj(USD/yr) | Delta Obj | EENS(MWh/yr) | Delta EENS | Curt(MWh/yr)")
    for r in results:
        m = r["metrics"]
        delta_obj = m["objective"] - base_obj
        delta_eens = m["eens_mwh"] - base_eens
        print(
            f"{r['case']} | {r['termination']} | {m['objective']:,.2f} | {delta_obj:,.2f} | "
            f"{m['eens_mwh']:,.2f} | {delta_eens:,.2f} | {m['curtailment_mwh']:,.2f}"
        )

    return results


def solve_and_report():
    solver_name = _pick_solver()
    print(f"使用求解器: {solver_name}")
    result = _solve_single_case(
        case_name="Joint",
        allow_line=True,
        allow_storage=True,
        solver_name=solver_name,
        time_limit=90,
        mip_rel_gap=0.005,
        verbose=False,
    )
    model, tnep_data = build_model()
    _apply_case_investment_policy(model, allow_line=True, allow_storage=True)

    solver = _configure_solver(solver_name, time_limit=90, mip_rel_gap=0.005)
    result_raw = solver.solve(model)

    print("\n=== 求解状态 ===")
    print(result_raw.solver.termination_condition)
    print(f"目标函数值: {value(model.obj):,.2f}")

    selected_lines = [l for l in model.LC if value(model.y[l]) > 0.5]
    selected_storage = [h for h in model.H if value(model.z[h]) > 0.5]
    total_storage = sum(value(model.E[h]) for h in model.H)

    invest_line = sum(value(model.c_line[l]) * value(model.y[l]) for l in model.LC)
    invest_stor = sum(value(model.c_fix[h]) * value(model.z[h]) + value(model.c_cap) * value(model.E[h]) for h in model.H)
    shed_total = sum(
        value(model.omega[s]) * sum(value(model.d_shed[b, s, t]) * value(model.delta_t) for b in model.B for t in model.T)
        for s in model.S
    )
    curt_total = sum(
        value(model.omega[s]) * sum(
            (value(model.Wbar[r, s, t]) - value(model.w[r, s, t])) * value(model.delta_t)
            for r in model.R for t in model.T
        )
        for s in model.S
    )

    print("\n=== 投资决策 ===")
    print(f"已选候选线路数: {len(selected_lines)} / {len(list(model.LC))}")
    print(f"已选储能站点数: {len(selected_storage)} / {len(list(model.H))}")
    print(f"储能装机容量: {total_storage:.2f} MWh")
    print(f"线路年化投资: {invest_line:,.2f} USD/year")
    print(f"储能年化投资: {invest_stor:,.2f} USD/year")
    print(f"预算上限: {value(model.Gamma):,.2f} USD/year")

    print("\n=== 可靠性与新能源消纳 ===")
    print(f"年期望失供电量 (EENS): {shed_total:,.2f} MWh/year")
    print(f"年期望新能源弃电量: {curt_total:,.2f} MWh/year")

    print("\n已选线路（前20条）:")
    for l in selected_lines[:20]:
        print(f"  {l}, 成本={tnep_data['c_line'][l]:,.0f} USD/year")

    print("\n已选储能站点:")
    for h in selected_storage:
        print(f"  母线 {h}: E={value(model.E[h]):.2f} MWh")


if __name__ == "__main__":
    run_comparison_experiments()
