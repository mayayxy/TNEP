"""TNEP 场景数据生成与 CSV 读取（Representative Days）。"""

import csv
import os
import random


# ---------------- User switches ----------------
USE_SCENARIO_NOISE = True
NOISE_LEVEL = 0.05
NOISE_SEED = 2026
DELTA_T_HOUR = 4.0
DEMAND_BASE_SCALE = 0.92

CSV_DIR = os.path.join(os.path.dirname(__file__), "dataset_csv")


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _noise_multiplier(rng):
    return 1.0 + rng.uniform(-NOISE_LEVEL, NOISE_LEVEL)


def _read_required_rows(csv_dir, filename):
    fp = os.path.join(csv_dir, filename)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"缺少输入文件: {fp}")
    with open(fp, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_base_inputs(csv_dir):
    bus_rows = _read_required_rows(csv_dir, "buses.csv")
    _ = _read_required_rows(csv_dir, "generators.csv")
    ren_rows = _read_required_rows(csv_dir, "renewables.csv")

    base_load = {}
    for row in bus_rows:
        b = int(row["bus"])
        base_load[b] = _safe_float(row.get("base_load_mw", 0.0), 0.0)

    renewable_meta = {}
    for row in ren_rows:
        rid = row.get("ren_id", row.get("r"))
        if rid is None:
            raise ValueError("renewables.csv 缺少 ren_id/r 字段。")
        renewable_meta[rid] = {
            "bus": int(row["bus"]),
            "cap": _safe_float(row.get("capacity_mw", row.get("cap_mw", 0.0))),
            "type": row.get("type", "wind").strip().lower(),
        }
    return base_load, renewable_meta


def _scenario_config():
    scenarios = ["S1", "S2", "S3", "S4", "S5", "S6"]
    scenario_desc = {
        "S1": "Summer Peak",
        "S2": "Winter Peak",
        "S3": "Normal Day",
        "S4": "Weekend Low",
        "S5": "High Renewable",
        "S6": "Stress Day",
    }
    omega = {"S1": 60.0, "S2": 55.0, "S3": 120.0, "S4": 80.0, "S5": 30.0, "S6": 20.0}
    beta = {"S1": 1.30, "S2": 1.20, "S3": 1.00, "S4": 0.70, "S5": 0.90, "S6": 1.40}
    alpha = {1: 0.70, 2: 0.85, 3: 1.00, 4: 0.95, 5: 1.25, 6: 0.90}
    periods = [1, 2, 3, 4, 5, 6]

    if abs(sum(omega.values()) - 365.0) > 1e-9:
        raise ValueError("scenarios 权重和必须等于 365。")
    if alpha[5] != max(alpha.values()):
        raise ValueError("第 5 时段必须是系统峰值。")
    return scenarios, scenario_desc, omega, beta, alpha, periods


def _wind_cf_matrix():
    # 用户给定推荐 CF，按场景转置成 {scenario: {t: cf}}。
    raw = {
        1: {"S1": 0.25, "S2": 0.30, "S3": 0.28, "S4": 0.32, "S5": 0.40, "S6": 0.10},
        2: {"S1": 0.35, "S2": 0.40, "S3": 0.38, "S4": 0.42, "S5": 0.55, "S6": 0.15},
        3: {"S1": 0.60, "S2": 0.50, "S3": 0.55, "S4": 0.50, "S5": 0.85, "S6": 0.20},
        4: {"S1": 0.65, "S2": 0.55, "S3": 0.60, "S4": 0.55, "S5": 0.90, "S6": 0.25},
        5: {"S1": 0.40, "S2": 0.35, "S3": 0.38, "S4": 0.30, "S5": 0.70, "S6": 0.15},
        6: {"S1": 0.30, "S2": 0.25, "S3": 0.28, "S4": 0.25, "S5": 0.50, "S6": 0.10},
    }
    table = {s: {} for s in ["S1", "S2", "S3", "S4", "S5", "S6"]}
    for t, row in raw.items():
        for s, cf in row.items():
            table[s][t] = cf
    return table


def _solar_cf_from_wind(wind_cf):
    # 太阳能规律：t=1,6 为 0；t=3,4 高。
    factors = {1: 0.0, 2: 0.35, 3: 1.00, 4: 0.92, 5: 0.25, 6: 0.0}
    solar_cf = {}
    for s, t_map in wind_cf.items():
        solar_cf[s] = {}
        for t, cf in t_map.items():
            solar_cf[s][t] = max(0.0, min(1.0, cf * factors[t]))
    # S5 中午必须很高，强化弃风动力。
    solar_cf["S5"][3] = max(solar_cf["S5"][3], 0.90)
    solar_cf["S5"][4] = max(solar_cf["S5"][4], 0.88)
    return solar_cf


def _demand_space_multiplier(bus):
    # 空间不均匀性：负荷中心在 1-10，远端负荷弱化以制造潮流转移需求。
    if 1 <= bus <= 10:
        return 1.65
    if bus in (16, 21):
        return 0.30
    return 0.35


def _renewable_space_multiplier(bus):
    # 新能源集中在远端 16 / 21，其他位置弱化。
    return 1.45 if bus in (16, 21) else 0.60


def generate_representative_day_csvs(csv_dir):
    os.makedirs(csv_dir, exist_ok=True)
    base_load, renewable_meta = _read_base_inputs(csv_dir)
    scenarios, scenario_desc, omega, beta, alpha, periods = _scenario_config()
    wind_cf = _wind_cf_matrix()
    solar_cf = _solar_cf_from_wind(wind_cf)

    rng = random.Random(NOISE_SEED)
    period_labels = {
        1: "00:00-04:00",
        2: "04:00-08:00",
        3: "08:00-12:00",
        4: "12:00-16:00",
        5: "16:00-20:00",
        6: "20:00-24:00",
    }

    demand_rows = []
    peak_demand_mw = 0.0
    annual_demand_mwh = 0.0
    for bus in sorted(base_load):
        base_bus = max(0.0, base_load[bus]) * DEMAND_BASE_SCALE * _demand_space_multiplier(bus)
        for s in scenarios:
            for t in periods:
                val = base_bus * beta[s] * alpha[t]
                if s == "S6":
                    val *= 1.10
                if USE_SCENARIO_NOISE:
                    val *= _noise_multiplier(rng)
                val = max(0.0, val)
                demand_rows.append((bus, s, t, val))
                peak_demand_mw = max(peak_demand_mw, val)
                annual_demand_mwh += omega[s] * val * DELTA_T_HOUR

    avail_rows = []
    annual_renewable_mwh = 0.0
    for rid, meta in sorted(renewable_meta.items()):
        ren_type = meta["type"]
        bus = meta["bus"]
        cap = max(0.0, meta["cap"]) * _renewable_space_multiplier(bus)
        cf_table = solar_cf if ren_type == "solar" else wind_cf
        for s in scenarios:
            for t in periods:
                cf = cf_table[s][t]
                if s == "S5" and t in (3, 4):
                    cf += 0.05
                cf = max(0.0, min(1.0, cf))
                avail = cap * cf
                if USE_SCENARIO_NOISE:
                    avail *= _noise_multiplier(rng)
                avail = max(0.0, avail)
                avail_rows.append((rid, s, t, avail))
                annual_renewable_mwh += omega[s] * avail * DELTA_T_HOUR

    with open(os.path.join(csv_dir, "scenarios.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "description", "omega_days"])
        for s in scenarios:
            writer.writerow([s, scenario_desc[s], omega[s]])

    with open(os.path.join(csv_dir, "periods.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "time_range", "delta_hour"])
        for t in periods:
            writer.writerow([t, period_labels[t], DELTA_T_HOUR])

    # 兼容性检查要求：字段完全匹配。
    with open(os.path.join(csv_dir, "demand.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["bus", "scenario", "t", "demand_mw"])
        for row in demand_rows:
            writer.writerow(row)

    # 兼容性检查要求：字段完全匹配。
    with open(os.path.join(csv_dir, "renewable_availability.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["ren_id", "scenario", "t", "avail_mw"])
        for row in avail_rows:
            writer.writerow(row)

    print(f"Representative day 数据已写入: {csv_dir}")
    print(f"Total annual demand (MWh): {annual_demand_mwh:,.2f}")
    print(f"Peak demand (MW): {peak_demand_mw:,.2f}")
    print(f"Total renewable energy (MWh): {annual_renewable_mwh:,.2f}")


def load_tnep_data_from_csv(csv_dir):
    params = {}
    with open(os.path.join(csv_dir, "global_params.csv"), "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            params[row["parameter"]] = float(row["value"])

    buses = []
    with open(os.path.join(csv_dir, "buses.csv"), "r", newline="", encoding="utf-8-sig") as f:
        buses = [int(r["bus"]) for r in csv.DictReader(f)]

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

    periods = []
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

    assumptions = {
        k: v
        for k, v in params.items()
        if k not in {"base_mva", "c_cap", "c_shed", "c_curt", "Gamma", "rho", "eta_c", "eta_d", "delta_t"}
    }

    return {
        "buses": buses,
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
        "assumptions": assumptions,
    }


generate_representative_day_csvs(CSV_DIR)
TNEP_DATA = load_tnep_data_from_csv(CSV_DIR)


if __name__ == "__main__":
    print("RTS24 Representative Days 数据集已生成并读取")
    print("CSV目录:", CSV_DIR)
    print("母线数:", len(TNEP_DATA["buses"]))
    print("在役通道数:", len(TNEP_DATA["existing_lines"]))
    print("候选通道数:", len(TNEP_DATA["candidate_lines"]))
    print("场景数:", len(TNEP_DATA["scenarios"]), "时段数:", len(TNEP_DATA["periods"]))
    print("场景权重和:", sum(TNEP_DATA["omega"].values()))
    print("噪声开关:", USE_SCENARIO_NOISE)
