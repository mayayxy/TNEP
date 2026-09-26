"""投资候选启发式预筛选。"""

from collections import defaultdict


def build_bus_energy_stats(tnep_data):
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


def build_storage_scores(tnep_data):
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
        storage_scores.append((swing / max(1.0, cap_cost), h))
    storage_scores.sort(key=lambda x: x[0], reverse=True)
    return storage_scores


def select_heuristic_candidates(tnep_data, max_lines=10, max_storage=4, method="score"):
    if method == "all":
        lines = list(tnep_data["candidate_lines"])[: max(0, int(max_lines))]
        storage = list(tnep_data["storage_sites"])[: max(0, int(max_storage))]
        return lines, storage
    if method == "shed_union":
        return select_shed_union_candidates(tnep_data, max_lines=max_lines, max_storage=max_storage)
    load_bus, ren_bus, _ = build_bus_energy_stats(tnep_data)
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
            bridge = deficit[i] * surplus[j] + deficit[j] * surplus[i]
            score = (0.7 * transfer_potential + 0.2 * imbalance_gap + 0.1 * bridge**0.5) / c
        else:
            score = (transfer_potential + 0.5 * imbalance_gap) / c
        line_scores.append((score, l))
    line_scores.sort(key=lambda x: x[0], reverse=True)

    storage_scores = build_storage_scores(tnep_data)

    if method == "diversity":
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


def build_local_shed_by_bus(tnep_data):
    """不计网络：本地可再生+常规满发仍不能满足的年切负荷能量。"""
    gen_cap = defaultdict(float)
    for _g, meta in tnep_data["gen_meta"].items():
        gen_cap[meta["bus"]] += float(meta["pmax"])
    ren_ids = defaultdict(list)
    for rid, meta in tnep_data["renewable_meta"].items():
        ren_ids[meta["bus"]].append(rid)
    shed = defaultdict(float)
    for s in tnep_data["scenarios"]:
        w = tnep_data["omega"][s]
        for t in tnep_data["periods"]:
            for b in tnep_data["buses"]:
                demand = tnep_data["demand"][(b, s, t)]
                ren = sum(tnep_data["wbar"][(rid, s, t)] for rid in ren_ids[b])
                remain = demand - min(demand, ren) - gen_cap[b]
                shed[b] += w * max(0.0, remain) * tnep_data["delta_t"]
    return shed


def select_shed_union_candidates(tnep_data, max_lines=80, max_storage=16):
    """评分 Top-k 与高切负荷母线走廊取并，扩大池子以覆盖供电缺口通道。"""
    shed = build_local_shed_by_bus(tnep_data)
    score_lines, score_storage = select_heuristic_candidates(
        tnep_data, max_lines=max_lines, max_storage=max_storage, method="score"
    )
    line_shed_scores = []
    for l in tnep_data["candidate_lines"]:
        i, j = l
        c = max(1.0, tnep_data["c_line"][l])
        line_shed_scores.append(((shed[i] + shed[j]) / c, l))
    line_shed_scores.sort(key=lambda x: x[0], reverse=True)
    shed_lines = [l for _, l in line_shed_scores]

    selected_lines = []
    seen = set()
    for src in (score_lines, shed_lines):
        for l in src:
            if l in seen:
                continue
            selected_lines.append(l)
            seen.add(l)
            if len(selected_lines) >= max_lines:
                break
        if len(selected_lines) >= max_lines:
            break

    hot_buses = [b for b, v in sorted(shed.items(), key=lambda x: -x[1]) if v > 0]
    selected_storage = []
    seen_h = set()
    for h in hot_buses:
        if h in tnep_data["storage_sites"] and h not in seen_h:
            selected_storage.append(h)
            seen_h.add(h)
        if len(selected_storage) >= max_storage:
            break
    for h in score_storage:
        if h not in seen_h:
            selected_storage.append(h)
            seen_h.add(h)
        if len(selected_storage) >= max_storage:
            break
    return selected_lines, selected_storage


def rank_diversity_candidates(tnep_data):
    """按多样性规则给出全部候选线路/储能的入选顺序（不截断）。"""
    n_line = len(tnep_data["candidate_lines"])
    n_stor = len(tnep_data["storage_sites"])
    lines, storage = select_heuristic_candidates(
        tnep_data,
        max_lines=n_line,
        max_storage=n_stor,
        method="diversity",
    )
    score_lines, score_storage = select_heuristic_candidates(
        tnep_data,
        max_lines=n_line,
        max_storage=n_stor,
        method="score",
    )
    return {
        "diversity_lines": lines,
        "diversity_storage": storage,
        "score_lines": score_lines,
        "score_storage": score_storage,
    }


def apply_heuristic_fixings(
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
            model.y[l].fix(1 if l in forced_line_set else 0)
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
