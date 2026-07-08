"""从 CSV 目录加载 TNEP 建模数据。"""

import csv
import os


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
            if "base_kv" in r and r["base_kv"]:
                buses_kv[b] = float(r["base_kv"])
            elif b <= 10:
                buses_kv[b] = 138.0
            else:
                buses_kv[b] = 230.0

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
            gen_meta[g] = {
                "bus": int(r["bus"]),
                "pmax": float(r["pmax"]),
                "pmin": float(r["pmin"]),
                "c_gen": c_gen[g],
            }

    storage_sites, storage_meta = [], {}
    with open(os.path.join(csv_dir, "storage_sites.csv"), "r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            h = int(r["h"])
            storage_sites.append(h)
            storage_meta[h] = {
                "Ebar": float(r["Ebar_mwh"]),
                "c_fix": float(r["c_fix_usd_per_year"]),
            }

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
        if k
        not in {
            "base_mva",
            "c_cap",
            "c_shed",
            "c_curt",
            "Gamma",
            "rho",
            "eta_c",
            "eta_d",
            "delta_t",
        }
    }

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
        "assumptions": assumptions,
    }
