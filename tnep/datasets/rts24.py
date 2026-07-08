"""IEEE RTS-24 代表日时序数据生成。"""

import csv
import os
import random

USE_SCENARIO_NOISE = True
NOISE_LEVEL = 0.05
NOISE_SEED = 2026
DELTA_T_HOUR = 4.0
DEMAND_BASE_SCALE = 0.92


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

    base_load = {int(row["bus"]): _safe_float(row.get("base_load_mw", 0.0)) for row in bus_rows}
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
    return scenarios, scenario_desc, omega, beta, alpha, periods


def _wind_cf_matrix():
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
    factors = {1: 0.0, 2: 0.35, 3: 1.00, 4: 0.92, 5: 0.25, 6: 0.0}
    solar_cf = {}
    for s, t_map in wind_cf.items():
        solar_cf[s] = {t: max(0.0, min(1.0, cf * factors[t])) for t, cf in t_map.items()}
    solar_cf["S5"][3] = max(solar_cf["S5"][3], 0.90)
    solar_cf["S5"][4] = max(solar_cf["S5"][4], 0.88)
    return solar_cf


def _demand_space_multiplier(bus):
    if 1 <= bus <= 10:
        return 1.65
    if bus in (16, 21):
        return 0.30
    return 0.35


def _renewable_space_multiplier(bus):
    return 1.45 if bus in (16, 21) else 0.60


def build(out_dir):
    """生成 RTS-24 代表日时序 CSV（静态网络文件需已存在于 out_dir）。"""
    os.makedirs(out_dir, exist_ok=True)
    base_load, renewable_meta = _read_base_inputs(out_dir)
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
    annual_demand_mwh = 0.0
    peak_demand_mw = 0.0
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
        cf_table = solar_cf if meta["type"] == "solar" else wind_cf
        cap = max(0.0, meta["cap"]) * _renewable_space_multiplier(meta["bus"])
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

    with open(os.path.join(out_dir, "scenarios.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "description", "omega_days"])
        for s in scenarios:
            w.writerow([s, scenario_desc[s], omega[s]])

    with open(os.path.join(out_dir, "periods.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t", "time_range", "delta_hour"])
        for t in periods:
            w.writerow([t, period_labels[t], DELTA_T_HOUR])

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

    print(f"RTS-24 代表日数据已写入: {out_dir}")
    print(f"年总负荷 (MWh): {annual_demand_mwh:,.2f}")
    print(f"峰值负荷 (MW): {peak_demand_mw:,.2f}")
    print(f"年总新能源 (MWh): {annual_renewable_mwh:,.2f}")
