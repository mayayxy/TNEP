"""构建 IEEE 57-bus TNEP 数据集（独立目录，不覆盖现有 dataset_csv）。"""

from collections import deque
import csv
import itertools
import os
import random

from pypower.case57 import case57


USE_SCENARIO_NOISE = True
NOISE_LEVEL = 0.05
NOISE_SEED = 2026

DELTA_T_HOUR = 1.0
DEMAND_BASE_SCALE = 0.58
OUT_DIR = os.path.join(os.path.dirname(__file__), "dataset_csv_case57")


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


def build_case57_dataset(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    c = case57()

    bus_mat = c["bus"]
    branch_mat = c["branch"]
    gen_mat = c["gen"]
    gencost_mat = c["gencost"]

    buses = sorted(int(row[0]) for row in bus_mat)
    base_kv = {int(row[0]): float(row[9]) for row in bus_mat}
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

    max_graph_hops = 3
    pair_hops = all_pair_shortest_hops(buses, existing_set)

    candidate_lines = []
    for i, j in itertools.combinations(buses, 2):
        if (i, j) in existing_set:
            continue
        hops = pair_hops[(i, j)]
        if hops <= max_graph_hops:
            candidate_lines.append((i, j))
    ren_cluster = {49, 50, 52, 53, 55, 56, 57}
    top_load_set = set([b for b, _ in sorted(base_load.items(), key=lambda x: x[1], reverse=True)[:14]])
    scored = []
    for i, j in candidate_lines:
        score = 0
        if i in ren_cluster or j in ren_cluster:
            score += 2
        if i in top_load_set or j in top_load_set:
            score += 2
        score += max(0, 4 - pair_hops[(i, j)])
        scored.append((score, i, j))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    candidate_lines = [(i, j) for _, i, j in scored[:90]]

    line_crf = annualization_factor(0.05, 40)
    line_capex_per_mile = {138: 2_000_000.0, 230: 2_200_000.0, 500: 3_000_000.0}
    line_annual_cost_per_mile = {kv: line_capex_per_mile[kv] * line_crf for kv in line_capex_per_mile}

    def kv_level(i, j):
        k1, k2 = base_kv[i], base_kv[j]
        if k1 >= 400 or k2 >= 400:
            return 500
        if k1 >= 200 and k2 >= 200:
            return 230
        return 138

    def estimate_length_mile(hops):
        return 18.0 + 12.0 * max(0, int(hops) - 1)

    line_meta = {}
    for l in candidate_lines:
        i, j = l
        kv = kv_level(i, j)
        hops = pair_hops[l]
        length_mile = estimate_length_mile(hops)
        annual_cost = line_annual_cost_per_mile[kv] * length_mile
        line_meta[l] = {
            "kv": kv,
            "hops": hops,
            "length_mile": length_mile,
            "annual_cost_usd": annual_cost,
        }

    # 机组边际成本：二次成本曲线在中间出力点的导数。
    gen_meta = {}
    for idx, row in enumerate(gen_mat, start=1):
        bus = int(row[0])
        pmax = float(row[8])
        pmin = float(row[9])
        gc = gencost_mat[idx - 1]
        c2, c1 = float(gc[-3]), float(gc[-2])
        pmid = 0.5 * (pmax + pmin)
        mc = max(1.0, 2.0 * c2 * pmid + c1)
        gen_meta[idx] = {"bus": bus, "pmax": pmax, "pmin": pmin, "c_gen": mc}

    # 在 IEEE57 上外生配置风光资源，集中在远端母线制造送出压力。
    renewable_meta = {
        "R1": {"bus": 49, "cap": 220.0, "type": "wind"},
        "R2": {"bus": 50, "cap": 180.0, "type": "wind"},
        "R3": {"bus": 52, "cap": 140.0, "type": "solar"},
        "R4": {"bus": 53, "cap": 150.0, "type": "solar"},
        "R5": {"bus": 55, "cap": 200.0, "type": "wind"},
        "R6": {"bus": 56, "cap": 160.0, "type": "solar"},
        "R7": {"bus": 57, "cap": 210.0, "type": "wind"},
        "R8": {"bus": 44, "cap": 120.0, "type": "wind"},
        "R9": {"bus": 45, "cap": 130.0, "type": "solar"},
        "R10": {"bus": 46, "cap": 110.0, "type": "wind"},
    }

    # 选择高负荷母线作为储能候选点。
    top_load_buses = [b for b, _ in sorted(base_load.items(), key=lambda x: x[1], reverse=True)[:12]]
    storage_sites = sorted(top_load_buses)
    storage_meta = {h: {"Ebar": 400.0 if base_load[h] >= 60 else 280.0, "c_fix": 215_000.0} for h in storage_sites}

    # 场景与时段：12个代表日，24时段（1小时制）。
    scenarios = [f"S{i}" for i in range(1, 13)]
    omega = {"S1": 31, "S2": 30, "S3": 31, "S4": 30, "S5": 31, "S6": 30, "S7": 31, "S8": 30, "S9": 31, "S10": 30, "S11": 30, "S12": 30}
    if sum(omega.values()) != 365:
        raise ValueError("场景权重和必须等于365。")

    # 夏冬平日高负荷、周末低负荷、高新能源与极端压力。
    beta = {"S1": 1.20, "S2": 0.95, "S3": 1.10, "S4": 0.85, "S5": 1.05, "S6": 0.90, "S7": 1.00, "S8": 0.88, "S9": 1.25, "S10": 0.92, "S11": 0.98, "S12": 1.32}

    alpha_profile = [
        0.62, 0.58, 0.56, 0.55, 0.57, 0.63, 0.72, 0.84, 0.93, 0.98, 1.00, 0.97,
        0.96, 0.98, 1.04, 1.12, 1.20, 1.27, 1.30, 1.22, 1.08, 0.92, 0.79, 0.69,
    ]
    alpha = {t + 1: alpha_profile[t] for t in range(24)}
    if alpha[19] != max(alpha.values()):
        raise ValueError("第19时段应为系统峰值。")

    periods = list(range(1, 25))
    rng = random.Random(NOISE_SEED)

    # 负荷空间不均匀：高负荷中心强化，远端可再生区域负荷弱化。
    high_load_set = set(top_load_buses)
    ren_cluster = {49, 50, 52, 53, 55, 56, 57}

    demand_rows = []
    annual_demand_mwh = 0.0
    peak_demand = 0.0
    for b in buses:
        if b in high_load_set:
            spatial = 1.12
        elif b in ren_cluster:
            spatial = 0.74
        else:
            spatial = 0.82
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

    # 新能源可用率（风：全天非零；光：夜间为0，中午高）。
    wind_cf = {}
    solar_cf = {}
    for s in scenarios:
        wind_cf[s] = {}
        solar_cf[s] = {}
        for t in periods:
            if t <= 6:
                w = 0.28 + 0.03 * (t - 1)
            elif t <= 12:
                w = 0.35 + 0.02 * (t - 7)
            elif t <= 18:
                w = 0.45 - 0.015 * (t - 13)
            else:
                w = 0.30 - 0.01 * (t - 19)
            w = max(0.10, min(0.75, w))
            wind_cf[s][t] = w

            if 7 <= t <= 18:
                s_cf = 0.15 + 0.11 * (1.0 - abs(12.5 - t) / 6.5)
            else:
                s_cf = 0.0
            solar_cf[s][t] = max(0.0, min(0.9, s_cf))

    # 高新能源日提升可用率；极端压力日压低可用率。
    high_ren_days = {"S5", "S9"}
    stress_days = {"S12"}

    avail_rows = []
    annual_renewable_mwh = 0.0
    for rid, meta in renewable_meta.items():
        b = meta["bus"]
        cap = meta["cap"] * (1.35 if b in ren_cluster else 1.0)
        typ = meta["type"].lower()
        for s in scenarios:
            for t in periods:
                cf = solar_cf[s][t] if typ == "solar" else wind_cf[s][t]
                if s in high_ren_days and t in (11, 12, 13, 14):
                    cf += 0.08
                if s in stress_days:
                    cf *= 0.35
                cf = max(0.0, min(1.0, cf))
                avail = cap * cf
                if USE_SCENARIO_NOISE:
                    avail *= _noise_multiplier(rng)
                avail = max(0.0, avail)
                avail_rows.append((rid, s, t, avail))
                annual_renewable_mwh += omega[s] * avail * DELTA_T_HOUR

    # 成本与系统参数。
    c_cap = 247_000.0 * annualization_factor(0.08, 15)
    c_shed = 10_000.0
    c_curt = 80.0
    gamma = 75_000_000.0
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
        w.writerow(["use_scenario_noise", int(USE_SCENARIO_NOISE), ""])
        w.writerow(["noise_level", NOISE_LEVEL, ""])
        w.writerow(["noise_seed", NOISE_SEED, ""])
        w.writerow(["max_graph_hops", max_graph_hops, ""])

    with open(os.path.join(out_dir, "buses.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["bus", "base_kv", "base_load_mw"])
        for b in buses:
            w.writerow([b, base_kv[b], base_load[b]])

    with open(os.path.join(out_dir, "branches.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["from_bus", "to_bus", "status"])
        for i, j in existing_lines:
            w.writerow([i, j, 1])

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
        for rid, m in renewable_meta.items():
            w.writerow([rid, rid, m["bus"], m["cap"], m["type"]])

    with open(os.path.join(out_dir, "renewable_availability.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ren_id", "scenario", "t", "avail_mw"])
        for row in avail_rows:
            w.writerow(row)

    print(f"Case57 数据集已生成: {out_dir}")
    print("Buses:", len(buses))
    print("Existing lines:", len(existing_lines))
    print("Candidate lines:", len(candidate_lines))
    print("Scenarios:", len(scenarios), "Periods:", len(periods))
    print(f"Total annual demand (MWh): {annual_demand_mwh:,.2f}")
    print(f"Peak demand (MW): {peak_demand:,.2f}")
    print(f"Total renewable energy (MWh): {annual_renewable_mwh:,.2f}")


if __name__ == "__main__":
    build_case57_dataset(OUT_DIR)
