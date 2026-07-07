"""基于 RTS-GMLC 原始数据构建 TNEP 数据集（独立目录）。"""

from collections import defaultdict, deque
import csv
import datetime as dt
import itertools
import os
import random


USE_SCENARIO_NOISE = True
NOISE_LEVEL = 0.05
NOISE_SEED = 2026

DELTA_T_HOUR = 1.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "dataset_csv_rts_gmlc")
RTS_ROOT = os.path.join(os.path.dirname(__file__), "RTS-GMLC", "RTS_Data")


def annualization_factor(discount_rate, lifetime):
    r = float(discount_rate)
    n = int(lifetime)
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def _noise_multiplier(rng):
    return 1.0 + rng.uniform(-NOISE_LEVEL, NOISE_LEVEL)


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _read_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_timeseries(path):
    rows = _read_csv(path)
    data = defaultdict(dict)
    for r in rows:
        d = dt.date(int(r["Year"]), int(r["Month"]), int(r["Day"]))
        t = int(r["Period"])
        data[d][t] = r
    return data


def all_pair_shortest_hops(buses, existing_set):
    adj = {b: [] for b in buses}
    for i, j in existing_set:
        adj[i].append(j)
        adj[j].append(i)
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


def pick_day(candidates, key_fn, used, reverse=False):
    ranked = sorted(candidates, key=key_fn, reverse=reverse)
    for d in ranked:
        if d not in used:
            used.add(d)
            return d
    raise ValueError("没有可用日期可选。")


def pick_median_day(candidates, key_fn, used):
    ranked = sorted(candidates, key=key_fn)
    mid = len(ranked) // 2
    offsets = [0]
    for k in range(1, len(ranked)):
        offsets.extend([k, -k])
    for off in offsets:
        idx = mid + off
        if 0 <= idx < len(ranked):
            d = ranked[idx]
            if d not in used:
                used.add(d)
                return d
    raise ValueError("无法选取中位日期。")


def build_rts_gmlc_dataset(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(NOISE_SEED)

    src_bus = _read_csv(os.path.join(RTS_ROOT, "SourceData", "bus.csv"))
    src_branch = _read_csv(os.path.join(RTS_ROOT, "SourceData", "branch.csv"))
    src_gen = _read_csv(os.path.join(RTS_ROOT, "SourceData", "gen.csv"))

    ts_load = _load_timeseries(os.path.join(RTS_ROOT, "timeseries_data_files", "Load", "DAY_AHEAD_regional_Load.csv"))
    ts_wind = _load_timeseries(os.path.join(RTS_ROOT, "timeseries_data_files", "WIND", "DAY_AHEAD_wind.csv"))
    ts_pv = _load_timeseries(os.path.join(RTS_ROOT, "timeseries_data_files", "PV", "DAY_AHEAD_pv.csv"))
    ts_rtpv = _load_timeseries(os.path.join(RTS_ROOT, "timeseries_data_files", "RTPV", "DAY_AHEAD_rtpv.csv"))
    ts_csp = _load_timeseries(os.path.join(RTS_ROOT, "timeseries_data_files", "CSP", "DAY_AHEAD_Natural_Inflow.csv"))

    buses = sorted(int(r["Bus ID"]) for r in src_bus)
    base_kv = {int(r["Bus ID"]): _to_float(r["BaseKV"]) for r in src_bus}
    base_load = {int(r["Bus ID"]): max(0.0, _to_float(r["MW Load"])) for r in src_bus}
    bus_area = {int(r["Bus ID"]): int(float(r["Area"])) for r in src_bus}

    existing_set = set()
    for r in src_branch:
        i = int(r["From Bus"])
        j = int(r["To Bus"])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        existing_set.add((a, b))
    existing_lines = sorted(existing_set)

    # 常规机组与可再生机组拆分。
    ren_types = {"WIND", "PV", "RTPV", "CSP"}
    conventional_rows = []
    renewable_rows = []
    for r in src_gen:
        uid = r["GEN UID"].strip()
        typ = r["Unit Type"].strip().upper()
        pmax = _to_float(r["PMax MW"])
        if pmax <= 0:
            continue
        if typ in ren_types:
            renewable_rows.append(r)
        elif typ in {"SYNC_COND", "STORAGE"}:
            continue
        else:
            conventional_rows.append(r)

    # 可再生时序列名集合。
    wind_cols = set(next(iter(ts_wind.values()))[1].keys()) - {"Year", "Month", "Day", "Period"}
    pv_cols = set(next(iter(ts_pv.values()))[1].keys()) - {"Year", "Month", "Day", "Period"}
    rtpv_cols = set(next(iter(ts_rtpv.values()))[1].keys()) - {"Year", "Month", "Day", "Period"}
    csp_cols = set(next(iter(ts_csp.values()))[1].keys()) - {"Year", "Month", "Day", "Period"}

    renewable_meta = {}
    for r in renewable_rows:
        uid = r["GEN UID"].strip()
        typ = r["Unit Type"].strip().upper()
        if (typ == "WIND" and uid not in wind_cols) or (typ == "PV" and uid not in pv_cols) or (typ == "RTPV" and uid not in rtpv_cols) or (typ == "CSP" and uid not in csp_cols):
            continue
        renewable_meta[uid] = {
            "bus": int(r["Bus ID"]),
            "cap": _to_float(r["PMax MW"]),
            "type": typ.lower(),
            "source": typ,
        }

    gen_meta = {}
    for idx, r in enumerate(conventional_rows, start=1):
        pmax = _to_float(r["PMax MW"])
        pmin = _to_float(r["PMin MW"])
        hr0 = _to_float(r.get("HR_avg_0", 10000.0), 10000.0)
        fuel_price = _to_float(r.get("Fuel Price $/MMBTU", 3.0), 3.0)
        vom = _to_float(r.get("VOM", 0.0), 0.0)
        # HR_avg_0 约为 BTU/kWh，折算 MMBTU/MWh 需除以1000。
        c_gen = max(1.0, fuel_price * (hr0 / 1000.0) + vom)
        gen_meta[idx] = {
            "bus": int(r["Bus ID"]),
            "pmax": pmax,
            "pmin": pmin,
            "c_gen": c_gen,
        }

    # 以区域负荷序列构建日指标，并从全年挑选12个代表日。
    all_days = sorted(ts_load.keys())
    day_metrics = {}
    for d in all_days:
        load_energy = 0.0
        peak_load = 0.0
        ren_energy = 0.0
        wind_energy = 0.0
        solar_energy = 0.0
        max_ramp = 0.0
        last_load = None
        for t in range(1, 25):
            lr = ts_load[d][t]
            reg_load = _to_float(lr["1"]) + _to_float(lr["2"]) + _to_float(lr["3"])
            wr = ts_wind[d][t]
            pr = ts_pv[d][t]
            rr = ts_rtpv[d][t]
            cr = ts_csp[d][t]
            w = sum(_to_float(wr[k]) for k in wr if k not in {"Year", "Month", "Day", "Period"})
            pv = sum(_to_float(pr[k]) for k in pr if k not in {"Year", "Month", "Day", "Period"})
            rtpv = sum(_to_float(rr[k]) for k in rr if k not in {"Year", "Month", "Day", "Period"})
            csp = sum(_to_float(cr[k]) for k in cr if k not in {"Year", "Month", "Day", "Period"})
            ren = w + pv + rtpv + csp
            load_energy += reg_load
            ren_energy += ren
            wind_energy += w
            solar_energy += pv + rtpv + csp
            peak_load = max(peak_load, reg_load)
            if last_load is not None:
                max_ramp = max(max_ramp, abs(reg_load - last_load))
            last_load = reg_load
        ratio = ren_energy / max(load_energy, 1e-6)
        day_metrics[d] = {
            "load_energy": load_energy,
            "peak_load": peak_load,
            "ren_energy": ren_energy,
            "ren_ratio": ratio,
            "wind_energy": wind_energy,
            "solar_energy": solar_energy,
            "stress": peak_load * (1.0 - ratio),
            "ramp": max_ramp,
            "weekday": d.weekday() < 5,
        }

    used = set()
    summer = [d for d in all_days if d.month in (6, 7, 8)]
    winter = [d for d in all_days if d.month in (12, 1, 2)]
    spring = [d for d in all_days if d.month in (3, 4, 5)]
    autumn = [d for d in all_days if d.month in (9, 10, 11)]
    weekends = [d for d in all_days if not day_metrics[d]["weekday"]]
    weekdays = [d for d in all_days if day_metrics[d]["weekday"]]

    scenario_day = {}
    scenario_day["S1"] = pick_day(summer, lambda d: day_metrics[d]["peak_load"], used, reverse=True)       # Summer Peak
    scenario_day["S2"] = pick_day(winter, lambda d: day_metrics[d]["peak_load"], used, reverse=True)       # Winter Peak
    scenario_day["S3"] = pick_median_day(spring, lambda d: day_metrics[d]["load_energy"], used)            # Spring Normal
    scenario_day["S4"] = pick_day(weekends, lambda d: day_metrics[d]["load_energy"], used, reverse=False)  # Weekend Low
    scenario_day["S5"] = pick_day(all_days, lambda d: day_metrics[d]["ren_ratio"], used, reverse=True)     # High Renewable
    scenario_day["S6"] = pick_day(all_days, lambda d: day_metrics[d]["stress"], used, reverse=True)        # Stress Day
    scenario_day["S7"] = pick_median_day(autumn, lambda d: day_metrics[d]["load_energy"], used)            # Autumn Normal
    scenario_day["S8"] = pick_day(all_days, lambda d: day_metrics[d]["wind_energy"], used, reverse=True)   # High Wind
    scenario_day["S9"] = pick_day(all_days, lambda d: day_metrics[d]["solar_energy"], used, reverse=True)  # High Solar
    scenario_day["S10"] = pick_day(weekdays, lambda d: day_metrics[d]["load_energy"], used, reverse=False) # Shoulder Low
    scenario_day["S11"] = pick_day(winter, lambda d: day_metrics[d]["ren_ratio"], used, reverse=True)      # Winter High Ren
    scenario_day["S12"] = pick_day(summer, lambda d: day_metrics[d]["stress"], used, reverse=True)         # Summer Stress

    scenarios = [f"S{i}" for i in range(1, 13)]
    scenario_desc = {
        "S1": "Summer Peak",
        "S2": "Winter Peak",
        "S3": "Spring Normal",
        "S4": "Weekend Low",
        "S5": "High Renewable",
        "S6": "Stress Day",
        "S7": "Autumn Normal",
        "S8": "High Wind",
        "S9": "High Solar",
        "S10": "Shoulder Low",
        "S11": "Winter High Renewable",
        "S12": "Summer Stress",
    }
    omega = {"S1": 35, "S2": 35, "S3": 45, "S4": 52, "S5": 24, "S6": 20, "S7": 45, "S8": 30, "S9": 24, "S10": 25, "S11": 15, "S12": 15}
    if sum(omega.values()) != 365:
        raise ValueError("场景权重和必须等于365。")
    periods = list(range(1, 25))

    # 区域负荷按母线基础负荷比例分配。
    area_total = defaultdict(float)
    area_buses = defaultdict(list)
    for b in buses:
        a = bus_area[b]
        area_buses[a].append(b)
        area_total[a] += base_load[b]
    area_weights = {}
    for a, blist in area_buses.items():
        den = area_total[a]
        if den <= 1e-9:
            w = 1.0 / len(blist)
            area_weights[a] = {b: w for b in blist}
        else:
            area_weights[a] = {b: base_load[b] / den for b in blist}

    top_load_buses = [b for b, _ in sorted(base_load.items(), key=lambda x: x[1], reverse=True)[:18]]
    ren_bus_set = {meta["bus"] for meta in renewable_meta.values()}

    demand_rows = []
    annual_demand_mwh = 0.0
    peak_demand = 0.0
    for s in scenarios:
        d = scenario_day[s]
        for t in periods:
            lr = ts_load[d][t]
            reg_load = {1: _to_float(lr["1"]), 2: _to_float(lr["2"]), 3: _to_float(lr["3"])}
            for b in buses:
                val = reg_load[bus_area[b]] * area_weights[bus_area[b]][b]
                if b in top_load_buses:
                    val *= 1.12
                if b in ren_bus_set:
                    val *= 0.86
                if s in {"S6", "S12"}:
                    val *= 1.08
                if s in {"S5", "S8", "S9", "S11"}:
                    val *= 0.95
                if USE_SCENARIO_NOISE:
                    val *= _noise_multiplier(rng)
                val = max(0.0, val)
                demand_rows.append((b, s, t, val))
                annual_demand_mwh += omega[s] * val * DELTA_T_HOUR
                peak_demand = max(peak_demand, val)

    avail_rows = []
    annual_renewable_mwh = 0.0
    for rid, meta in renewable_meta.items():
        typ = meta["source"]
        for s in scenarios:
            d = scenario_day[s]
            for t in periods:
                if typ == "WIND":
                    raw = _to_float(ts_wind[d][t].get(rid, 0.0))
                elif typ == "PV":
                    raw = _to_float(ts_pv[d][t].get(rid, 0.0))
                elif typ == "RTPV":
                    raw = _to_float(ts_rtpv[d][t].get(rid, 0.0))
                else:
                    raw = _to_float(ts_csp[d][t].get(rid, 0.0))

                val = raw
                if s in {"S5", "S9", "S11"} and 11 <= t <= 15:
                    val *= 1.10
                if s in {"S6", "S12"}:
                    val *= 0.60
                if USE_SCENARIO_NOISE:
                    val *= _noise_multiplier(rng)
                val = max(0.0, min(val, meta["cap"] * 1.05))
                avail_rows.append((rid, s, t, val))
                annual_renewable_mwh += omega[s] * val * DELTA_T_HOUR

    # 候选线路：按图距离与“负荷中心-新能源中心”优先打分。
    pair_hops = all_pair_shortest_hops(buses, existing_set)
    high_load_set = set(top_load_buses[:14])
    candidate_all = []
    for i, j in itertools.combinations(buses, 2):
        if (i, j) in existing_set:
            continue
        hops = pair_hops[(i, j)]
        if hops <= 3:
            score = 0
            if (i in high_load_set and j in ren_bus_set) or (j in high_load_set and i in ren_bus_set):
                score += 4
            if i in ren_bus_set or j in ren_bus_set:
                score += 2
            if i in high_load_set or j in high_load_set:
                score += 1
            score += max(0, 4 - hops)
            candidate_all.append((score, i, j))
    candidate_all.sort(key=lambda x: (-x[0], x[1], x[2]))
    candidate_lines = [(i, j) for _, i, j in candidate_all[:20]]

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

    # 储能候选点：高负荷点 + 新能源汇集点。
    storage_seed = list(dict.fromkeys(top_load_buses[:12] + sorted(ren_bus_set)[:6]))
    storage_sites = sorted(storage_seed[:6])
    storage_meta = {}
    for h in storage_sites:
        if h in high_load_set:
            ebar = 500.0
        elif h in ren_bus_set:
            ebar = 420.0
        else:
            ebar = 320.0
        storage_meta[h] = {"Ebar": ebar, "c_fix": 230_000.0}

    c_cap = 247_000.0 * annualization_factor(0.08, 15)
    c_shed = 10_000.0
    c_curt = 90.0
    gamma = 130_000_000.0
    rho, eta_c, eta_d = 0.5, 0.922, 0.922

    # 写出CSV。
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

    with open(os.path.join(out_dir, "renewables.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["r", "ren_id", "bus", "capacity_mw", "type"])
        for rid in sorted(renewable_meta):
            m = renewable_meta[rid]
            w.writerow([rid, rid, m["bus"], m["cap"], m["type"]])

    with open(os.path.join(out_dir, "storage_sites.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["h", "Ebar_mwh", "c_fix_usd_per_year"])
        for h in storage_sites:
            m = storage_meta[h]
            w.writerow([h, m["Ebar"], m["c_fix"]])

    with open(os.path.join(out_dir, "scenarios.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "description", "date", "omega_days"])
        for s in scenarios:
            w.writerow([s, scenario_desc[s], scenario_day[s].isoformat(), omega[s]])

    with open(os.path.join(out_dir, "periods.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t", "time_range", "delta_hour"])
        for t in periods:
            h0 = t - 1
            h1 = t
            w.writerow([t, f"{h0:02d}:00-{h1:02d}:00", DELTA_T_HOUR])

    with open(os.path.join(out_dir, "demand.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["bus", "scenario", "t", "demand_mw"])
        for row in demand_rows:
            w.writerow(row)

    with open(os.path.join(out_dir, "renewable_availability.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ren_id", "scenario", "t", "avail_mw"])
        for row in avail_rows:
            w.writerow(row)

    print(f"RTS-GMLC 数据集已生成: {out_dir}")
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
    build_rts_gmlc_dataset(OUT_DIR)
