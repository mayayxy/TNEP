"""IEEE 300-bus 数据集构建。"""

from collections import deque
import csv
import itertools
import os
import random

from pypower.case300 import case300


USE_SCENARIO_NOISE = True
NOISE_LEVEL = 0.05
NOISE_SEED = 2026

DELTA_T_HOUR = 1.0
DEMAND_BASE_SCALE = 0.72


def annualization_factor(discount_rate, lifetime):
    r = float(discount_rate)
    n = int(lifetime)
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def _noise_multiplier(rng):
    return 1.0 + rng.uniform(-NOISE_LEVEL, NOISE_LEVEL)


def all_pair_shortest_hops(buses, existing_set):
    adj = {b: [] for b in buses}
    for a, b in existing_set:
        adj[a].append(b)
        adj[b].append(a)
    hops = {}
    for s in buses:
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for t in buses:
            if s < t:
                hops[(s, t)] = dist.get(t, 999)
    return hops


def _marginal_cost_from_gencost(gencost_row, pmin, pmax):
    """从 MATPOWER gencost 估算机组边际成本（USD/MWh）。"""
    model = int(gencost_row[0])
    ncost = int(gencost_row[3])
    coeffs = list(gencost_row[4 : 4 + ncost]) if ncost > 0 else []
    pmid = 0.5 * (pmin + pmax)

    # model=2: 多项式成本，系数顺序是 [c_n ... c1 c0]。
    if model == 2 and ncost >= 2:
        if ncost >= 3:
            c2 = float(coeffs[-3])
            c1 = float(coeffs[-2])
            return max(1.0, 2.0 * c2 * pmid + c1)
        c1 = float(coeffs[-2])
        return max(1.0, c1)

    # 其他模型或异常情况，回退到保守常数。
    return 30.0


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(NOISE_SEED)

    c = case300()
    bus_mat = c["bus"]
    branch_mat = c["branch"]
    gen_mat = c["gen"]
    gencost_mat = c["gencost"]

    buses = sorted(int(row[0]) for row in bus_mat)
    base_kv = {int(row[0]): (float(row[9]) if float(row[9]) > 0 else 230.0) for row in bus_mat}
    base_load = {int(row[0]): max(0.0, float(row[2])) for row in bus_mat}

    existing_set = set()
    for row in branch_mat:
        i = int(row[0])
        j = int(row[1])
        status = int(row[10])
        if status != 1 or i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        existing_set.add((a, b))
    existing_lines = sorted(existing_set)

    pair_hops = all_pair_shortest_hops(buses, existing_set)
    max_graph_hops = 3
    high_load_set = {b for b, _ in sorted(base_load.items(), key=lambda x: x[1], reverse=True)[:80]}

    candidate_pool = []
    for i, j in itertools.combinations(buses, 2):
        if (i, j) in existing_set:
            continue
        hops = pair_hops[(i, j)]
        if hops > max_graph_hops:
            continue
        score = max(0, 4 - hops)
        if i in high_load_set or j in high_load_set:
            score += 2
        if i in high_load_set and j in high_load_set:
            score += 2
        candidate_pool.append((score, i, j))
    candidate_pool.sort(key=lambda x: (-x[0], x[1], x[2]))
    candidate_lines = [(i, j) for _, i, j in candidate_pool[:260]]

    line_crf = annualization_factor(0.05, 40)
    line_capex_per_mile = {138: 2_000_000.0, 230: 2_250_000.0, 500: 3_100_000.0}
    line_annual_cost_per_mile = {kv: line_capex_per_mile[kv] * line_crf for kv in line_capex_per_mile}

    def kv_level(i, j):
        k1, k2 = base_kv[i], base_kv[j]
        if k1 >= 400 or k2 >= 400:
            return 500
        if k1 >= 200 and k2 >= 200:
            return 230
        return 138

    line_meta = {}
    for i, j in candidate_lines:
        hops = pair_hops[(i, j)]
        kv = kv_level(i, j)
        length_mile = 18.0 + 12.0 * max(0, int(hops) - 1)
        annual_cost = line_annual_cost_per_mile[kv] * length_mile
        line_meta[(i, j)] = {
            "kv": kv,
            "hops": hops,
            "length_mile": length_mile,
            "annual_cost_usd": annual_cost,
        }

    gen_meta = {}
    for idx, row in enumerate(gen_mat, start=1):
        pmax = float(row[8])
        pmin = float(row[9])
        if pmax <= 0:
            continue
        gc = gencost_mat[min(idx - 1, len(gencost_mat) - 1)]
        c_gen = _marginal_cost_from_gencost(gc, pmin, pmax)
        gen_meta[idx] = {
            "bus": int(row[0]),
            "pmax": pmax,
            "pmin": max(0.0, pmin),
            "c_gen": c_gen,
        }

    gen_bus_set = {m["bus"] for m in gen_meta.values()}
    load_ranked = [b for b, _ in sorted(base_load.items(), key=lambda x: x[1], reverse=True)]
    renewable_buses = [b for b in load_ranked if b not in gen_bus_set][:40]
    if len(renewable_buses) < 40:
        renewable_buses = load_ranked[:40]

    renewable_meta = {}
    for idx, b in enumerate(renewable_buses, start=1):
        typ = "wind" if idx % 2 else "solar"
        base_cap = 90.0 + 2.5 * (idx % 10)
        if b in high_load_set:
            base_cap *= 0.9
        renewable_meta[f"R{idx}"] = {"bus": b, "cap": base_cap, "type": typ}

    storage_sites = sorted(load_ranked[:30])
    storage_meta = {}
    for h in storage_sites:
        ebar = 700.0 if h in high_load_set else 520.0
        storage_meta[h] = {"Ebar": ebar, "c_fix": 260_000.0}

    scenarios = [f"S{i}" for i in range(1, 13)]
    omega = {
        "S1": 31,
        "S2": 30,
        "S3": 31,
        "S4": 30,
        "S5": 31,
        "S6": 30,
        "S7": 31,
        "S8": 30,
        "S9": 31,
        "S10": 30,
        "S11": 30,
        "S12": 30,
    }
    periods = list(range(1, 25))

    beta = {
        "S1": 1.18,
        "S2": 0.96,
        "S3": 1.08,
        "S4": 0.84,
        "S5": 1.03,
        "S6": 0.90,
        "S7": 1.00,
        "S8": 0.88,
        "S9": 1.22,
        "S10": 0.92,
        "S11": 0.98,
        "S12": 1.30,
    }
    alpha_profile = [
        0.62,
        0.58,
        0.56,
        0.55,
        0.57,
        0.63,
        0.72,
        0.84,
        0.93,
        0.98,
        1.00,
        0.97,
        0.96,
        0.98,
        1.04,
        1.12,
        1.20,
        1.27,
        1.30,
        1.22,
        1.08,
        0.92,
        0.79,
        0.69,
    ]
    alpha = {t + 1: alpha_profile[t] for t in range(24)}

    demand_rows = []
    annual_demand_mwh = 0.0
    peak_demand = 0.0
    for b in buses:
        spatial = 1.10 if b in high_load_set else 0.88
        base_b = base_load[b] * spatial * DEMAND_BASE_SCALE
        for s in scenarios:
            for t in periods:
                val = base_b * beta[s] * alpha[t]
                if s == "S12":
                    val *= 1.05
                if USE_SCENARIO_NOISE:
                    val *= _noise_multiplier(rng)
                val = max(0.0, val)
                demand_rows.append((b, s, t, val))
                annual_demand_mwh += omega[s] * val * DELTA_T_HOUR
                peak_demand = max(peak_demand, val)

    wind_cf = {}
    solar_cf = {}
    for s in scenarios:
        wind_cf[s] = {}
        solar_cf[s] = {}
        for t in periods:
            if t <= 6:
                w = 0.30 + 0.025 * (t - 1)
            elif t <= 12:
                w = 0.40 + 0.015 * (t - 7)
            elif t <= 18:
                w = 0.49 - 0.013 * (t - 13)
            else:
                w = 0.33 - 0.008 * (t - 19)
            wind_cf[s][t] = max(0.10, min(0.75, w))

            if 7 <= t <= 18:
                s_cf = 0.16 + 0.12 * (1.0 - abs(12.5 - t) / 6.5)
            else:
                s_cf = 0.0
            solar_cf[s][t] = max(0.0, min(0.9, s_cf))

    avail_rows = []
    annual_renewable_mwh = 0.0
    for rid, meta in renewable_meta.items():
        cap = meta["cap"] * 1.4
        typ = meta["type"]
        for s in scenarios:
            for t in periods:
                cf = solar_cf[s][t] if typ == "solar" else wind_cf[s][t]
                if s in {"S5", "S9"} and 11 <= t <= 14:
                    cf += 0.08
                if s == "S12":
                    cf *= 0.35
                cf = max(0.0, min(1.0, cf))
                avail = cap * cf
                if USE_SCENARIO_NOISE:
                    avail *= _noise_multiplier(rng)
                avail = max(0.0, avail)
                avail_rows.append((rid, s, t, avail))
                annual_renewable_mwh += omega[s] * avail * DELTA_T_HOUR

    c_cap = 247_000.0 * annualization_factor(0.08, 15)
    c_shed = 10_000.0
    c_curt = 90.0
    gamma = 280_000_000.0
    rho, eta_c, eta_d = 0.5, 0.922, 0.922

    with open(os.path.join(out_dir, "global_params.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "unit"])
        w.writerow(["base_mva", 100.0, "MVA"])
        w.writerow(["c_cap", c_cap, "USD/MWh-year"])
        w.writerow(["c_shed", c_shed, "USD/MWh"])
        w.writerow(["c_curt", c_curt, "USD/MWh"])
        w.writerow(["Gamma", gamma, "USD/year"])
        w.writerow(["rho", rho, "MW/MWh"])
        w.writerow(["eta_c", eta_c, ""])
        w.writerow(["eta_d", eta_d, ""])
        w.writerow(["delta_t", DELTA_T_HOUR, "hour"])

    with open(os.path.join(out_dir, "buses.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["bus", "base_kv", "base_load_mw"])
        for b in buses:
            w.writerow([b, base_kv[b], base_load[b]])

    with open(os.path.join(out_dir, "existing_lines.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["from_bus", "to_bus"])
        for i, j in existing_lines:
            w.writerow([i, j])

    with open(os.path.join(out_dir, "candidate_lines.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["from_bus", "to_bus", "kv", "hops", "length_mile", "c_line_usd_per_year"])
        for i, j in candidate_lines:
            m = line_meta[(i, j)]
            w.writerow([i, j, m["kv"], m["hops"], m["length_mile"], m["annual_cost_usd"]])

    with open(os.path.join(out_dir, "generators.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["g", "bus", "pmax", "pmin", "c_gen_usd_per_mwh"])
        for g in sorted(gen_meta):
            m = gen_meta[g]
            w.writerow([g, m["bus"], m["pmax"], m["pmin"], m["c_gen"]])

    with open(os.path.join(out_dir, "storage_sites.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["h", "Ebar_mwh", "c_fix_usd_per_year"])
        for h in storage_sites:
            m = storage_meta[h]
            w.writerow([h, m["Ebar"], m["c_fix"]])

    with open(os.path.join(out_dir, "scenarios.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "omega_days"])
        for s in scenarios:
            w.writerow([s, omega[s]])

    with open(os.path.join(out_dir, "periods.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t", "time_range"])
        for t in periods:
            h0 = t - 1
            h1 = t
            w.writerow([t, f"{h0:02d}:00-{h1:02d}:00"])

    with open(os.path.join(out_dir, "demand.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["bus", "scenario", "t", "demand_mw"])
        for row in demand_rows:
            w.writerow(row)

    with open(os.path.join(out_dir, "renewables.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["r", "ren_id", "bus", "capacity_mw", "type"])
        for rid in sorted(renewable_meta):
            m = renewable_meta[rid]
            w.writerow([rid, rid, m["bus"], m["cap"], m["type"]])

    with open(os.path.join(out_dir, "renewable_availability.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ren_id", "scenario", "t", "avail_mw"])
        for row in avail_rows:
            w.writerow(row)

    print(f"Case300 数据集已生成: {out_dir}")
    print("Buses:", len(buses))
    print("Existing lines:", len(existing_lines))
    print("Candidate lines:", len(candidate_lines))
    print("Conventional generators:", len(gen_meta))
    print("Renewables:", len(renewable_meta))
    print("Storage sites:", len(storage_sites))
    print("Scenarios:", len(scenarios), "Periods:", len(periods))
    print(f"Total annual demand (MWh): {annual_demand_mwh:,.2f}")
    print(f"Peak demand (MW): {peak_demand:,.2f}")
    print(f"Total renewable energy (MWh): {annual_renewable_mwh:,.2f}")


if __name__ == "__main__":
    from tnep.config import dataset_dir

    build(dataset_dir("case300"))
