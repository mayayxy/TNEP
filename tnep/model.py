"""Pyomo MILP 模型构建。"""

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Reals,
    Set,
    Var,
    minimize,
)

from tnep.io import load_tnep_data_from_csv


def infer_b_ref(buses):
    if 13 in buses:
        return 13
    return min(buses)


def infer_line_electric_params(tnep_data):
    buses_kv = tnep_data["buses_kv"]
    lines = tnep_data["existing_lines"] + tnep_data["candidate_lines"]

    def kv_level(i, j):
        k1, k2 = buses_kv[i], buses_kv[j]
        if k1 >= 400 or k2 >= 400:
            return 500
        if k1 >= 200 and k2 >= 200:
            return 230
        if k1 < 200 and k2 < 200:
            return 138
        return 161

    fbar, b = {}, {}
    for i, j in lines:
        kv = kv_level(i, j)
        if kv == 500:
            x, f = 0.035, 900.0
        elif kv == 230:
            x, f = 0.06, 500.0
        elif kv == 138:
            x, f = 0.12, 200.0 if max(buses_kv.values()) > 300 else 175.0
        else:
            x, f = 0.08, 400.0
        b[(i, j)] = 1.0 / x
        fbar[(i, j)] = f
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
        "b_ref": infer_b_ref(tnep_data["buses"]),
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
    m.M = Param(
        m.LC,
        initialize={l: b_param[l] * op["theta_max"] * tnep_data["base_mva"] for l in tnep_data["candidate_lines"]},
    )

    m.Pmax = Param(m.G, initialize={g: meta["pmax"] for g, meta in tnep_data["gen_meta"].items()})
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
        invest = sum(model.c_line[l] * model.y[l] for l in model.LC) + sum(
            model.c_fix[h] * model.z[h] + model.c_cap * model.E[h] for h in model.H
        )
        op_cost = sum(
            model.omega[s]
            * sum(
                (
                    sum(model.c_gen[g] * model.p[g, s, t] * model.delta_t for g in model.G)
                    + model.c_shed * sum(model.d_shed[b, s, t] * model.delta_t for b in model.B)
                    + model.c_curt
                    * sum((model.Wbar[r, s, t] - model.w[r, s, t]) * model.delta_t for r in model.R)
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
    m.dc_exist = Constraint(
        m.L0,
        m.S,
        m.T,
        rule=lambda model, i, j, s, t: model.f[(i, j), s, t]
        == model.Bline[(i, j)] * (model.theta[i, s, t] - model.theta[j, s, t]),
    )
    m.dc_cand_up = Constraint(
        m.LC,
        m.S,
        m.T,
        rule=lambda model, i, j, s, t: model.f[(i, j), s, t]
        - model.Bline[(i, j)] * (model.theta[i, s, t] - model.theta[j, s, t])
        <= model.M[(i, j)] * (1 - model.y[(i, j)]),
    )
    m.dc_cand_dn = Constraint(
        m.LC,
        m.S,
        m.T,
        rule=lambda model, i, j, s, t: model.f[(i, j), s, t]
        - model.Bline[(i, j)] * (model.theta[i, s, t] - model.theta[j, s, t])
        >= -model.M[(i, j)] * (1 - model.y[(i, j)]),
    )
    m.thermal_exist_up = Constraint(
        m.L0, m.S, m.T, rule=lambda model, i, j, s, t: model.f[(i, j), s, t] <= model.Fbar[(i, j)]
    )
    m.thermal_exist_dn = Constraint(
        m.L0, m.S, m.T, rule=lambda model, i, j, s, t: model.f[(i, j), s, t] >= -model.Fbar[(i, j)]
    )
    m.thermal_cand_up = Constraint(
        m.LC, m.S, m.T, rule=lambda model, i, j, s, t: model.f[(i, j), s, t] <= model.Fbar[(i, j)] * model.y[(i, j)]
    )
    m.thermal_cand_dn = Constraint(
        m.LC, m.S, m.T, rule=lambda model, i, j, s, t: model.f[(i, j), s, t] >= -model.Fbar[(i, j)] * model.y[(i, j)]
    )
    m.gen_bound = Constraint(m.G, m.S, m.T, rule=lambda model, g, s, t: model.p[g, s, t] <= model.Pmax[g])
    m.ren_bound = Constraint(m.R, m.S, m.T, rule=lambda model, r, s, t: model.w[r, s, t] <= model.Wbar[r, s, t])
    m.stor_discharge = Constraint(m.H, m.S, m.T, rule=lambda model, h, s, t: model.e_plus[h, s, t] <= model.rho * model.E[h])
    m.stor_charge = Constraint(m.H, m.S, m.T, rule=lambda model, h, s, t: model.e_minus[h, s, t] <= model.rho * model.E[h])
    m.soc_dyn = Constraint(
        m.H,
        m.S,
        m.T,
        rule=lambda model, h, s, t: model.q[h, s, t]
        == model.q[h, s, t - 1]
        + model.eta_c * model.e_minus[h, s, t] * model.delta_t
        - (model.e_plus[h, s, t] / model.eta_d) * model.delta_t,
    )
    m.soc_upper = Constraint(m.H, m.S, m.T0, rule=lambda model, h, s, t: model.q[h, s, t] <= model.E[h])
    m.soc_cyclic = Constraint(m.H, m.S, rule=lambda model, h, s: model.q[h, s, 0] == model.q[h, s, max(model.T)])
    m.stor_link = Constraint(m.H, rule=lambda model, h: model.E[h] <= model.Ebar[h] * model.z[h])
    m.budget = Constraint(
        rule=lambda model: sum(model.c_line[l] * model.y[l] for l in model.LC)
        + sum(model.c_fix[h] * model.z[h] + model.c_cap * model.E[h] for h in model.H)
        <= model.Gamma
    )
    m.shed_bound = Constraint(m.B, m.S, m.T, rule=lambda model, b, s, t: model.d_shed[b, s, t] <= model.D[b, s, t])
    m.ref_bus = Constraint(m.S, m.T, rule=lambda model, s, t: model.theta[model.b_ref, s, t] == 0.0)
    m.ang_diff_up = Constraint(
        m.L, m.S, m.T, rule=lambda model, i, j, s, t: model.theta[i, s, t] - model.theta[j, s, t] <= model.theta_max
    )
    m.ang_diff_dn = Constraint(
        m.L, m.S, m.T, rule=lambda model, i, j, s, t: model.theta[i, s, t] - model.theta[j, s, t] >= -model.theta_max
    )
    return m, tnep_data
