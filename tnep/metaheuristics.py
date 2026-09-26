"""投资层元启发式：遗传算法（GA）与禁忌搜索（Tabu）。

染色体为候选池上的线路/储能 0-1 向量；适应度通过固定投资后求解运行层得到。
搜索可用精简代表日，结束时在全部代表日上复评。
"""

from __future__ import annotations

import json
import math
import os
import random
import time

from tnep.alns import (
    build_candidate_pool,
    evaluate_invest,
    invest_cost_min,
    make_eval_solver,
    make_initial_solution,
    _line_cost,
    _storage_cap_cost,
    _storage_fixed_cost,
)
from tnep.model import build_model
from tnep.solver import configure_solver, pick_solver


def _repair(lines, storage, pool, data, max_lines, max_storage):
    lines = set(lines)
    storage = set(storage)
    budget = float(data["Gamma"])

    def used():
        return invest_cost_min(data, lines, storage)

    while len(lines) > max_lines:
        worst = min(lines, key=lambda l: pool["line_pref"].get(l, 0.0))
        lines.remove(worst)
    while len(storage) > max_storage:
        worst = min(storage, key=lambda h: pool["stor_pref"].get(h, 0.0))
        storage.remove(worst)
    while used() > budget and (lines or storage):
        if lines and (not storage or random.random() < 0.6):
            worst = max(lines, key=lambda l: _line_cost(data, l) / max(pool["line_pref"].get(l, 1e-6), 1e-6))
            lines.remove(worst)
        elif storage:
            worst = max(
                storage,
                key=lambda h: (_storage_fixed_cost(data, h) + _storage_cap_cost(data, h))
                / max(pool["stor_pref"].get(h, 1e-6), 1e-6),
            )
            storage.remove(worst)
        else:
            break
    return lines, storage


def _random_individual(pool, data, rng, max_lines, max_storage):
    n_l = rng.randint(max(1, max_lines // 3), max_lines)
    n_h = rng.randint(max(1, max_storage // 3), max_storage)
    lines = set(rng.sample(pool["lines"], min(n_l, len(pool["lines"]))))
    storage = set(rng.sample(pool["storage"], min(n_h, len(pool["storage"]))))
    return _repair(lines, storage, pool, data, max_lines, max_storage)


def _eval(eval_model, eval_data, solver, lines, storage, cache, eval_time):
    return evaluate_invest(eval_model, eval_data, solver, lines, storage, cache, time_limit=eval_time)


def run_ga(
    csv_dir,
    time_limit=900,
    pool_lines=60,
    pool_storage=16,
    pool_method="score",
    max_lines=22,
    max_storage=12,
    pop_size=12,
    seed=1,
    solver_preference="gurobi",
    eval_time_limit=90,
    lite_scenarios=None,
):
    rng = random.Random(seed)
    solver_name = pick_solver(preferred=solver_preference)
    eval_model, eval_data = build_model(csv_dir, complementarity=False)
    full_model, full_data = eval_model, eval_data
    pool = build_candidate_pool(full_data, pool_lines, pool_storage, method=pool_method)
    print("GA 评估: 全部代表日，运行层为连续松弛（固定投资后为 LP）")

    solver = make_eval_solver(eval_model, solver_name, time_limit=eval_time_limit, mip_rel_gap=0.01)
    cache = {}
    t_end = time.perf_counter() + float(time_limit)

    print(
        f"GA: pool={len(pool['lines'])}L/{len(pool['storage'])}H ({pool_method}), "
        f"cap={max_lines}L/{max_storage}H, pop={pop_size}, time_limit={time_limit}s"
    )

    pop = []
    seed_l, seed_s = make_initial_solution(pool, eval_data, max_lines=min(19, max_lines), max_storage=min(10, max_storage))
    pop.append(_eval(eval_model, eval_data, solver, seed_l, seed_s, cache, eval_time_limit))
    while len(pop) < pop_size and time.perf_counter() < t_end:
        l, s = _random_individual(pool, eval_data, rng, max_lines, max_storage)
        pop.append(_eval(eval_model, eval_data, solver, l, s, cache, eval_time_limit))

    pop.sort(key=lambda x: x.obj)
    best = pop[0]
    gen = 0
    print(f"  gen=0 best={best.obj:,.2f} lines={len(best.lines)} stor={len(best.storage)} cache={len(cache)}")

    while time.perf_counter() < t_end:
        gen += 1
        # 锦标赛选父母
        def pick():
            a, b = rng.sample(pop, 2)
            return a if a.obj <= b.obj else b

        p1, p2 = pick(), pick()
        child_l = set()
        for l in pool["lines"]:
            src = p1 if rng.random() < 0.5 else p2
            if l in src.lines:
                child_l.add(l)
        child_s = set()
        for h in pool["storage"]:
            src = p1 if rng.random() < 0.5 else p2
            if h in src.storage:
                child_s.add(h)
        # 变异：随机翻转 1～3 条线、0～2 个站
        for _ in range(rng.randint(1, 3)):
            l = rng.choice(pool["lines"])
            if l in child_l:
                child_l.remove(l)
            else:
                child_l.add(l)
        for _ in range(rng.randint(0, 2)):
            h = rng.choice(pool["storage"])
            if h in child_s:
                child_s.remove(h)
            else:
                child_s.add(h)
        child_l, child_s = _repair(child_l, child_s, pool, eval_data, max_lines, max_storage)
        child = _eval(eval_model, eval_data, solver, child_l, child_s, cache, eval_time_limit)
        pop.append(child)
        pop.sort(key=lambda x: x.obj)
        pop = pop[:pop_size]
        if pop[0].obj < best.obj - 1e-6:
            best = pop[0]
        if gen % 5 == 0:
            print(
                f"  gen={gen} best={best.obj:,.2f} pop0={pop[0].obj:,.2f} "
                f"cache={len(cache)} remain={t_end - time.perf_counter():.0f}s"
            )

    return _finalize(full_model, full_data, solver_name, best, cache, gen, "GA", eval_time_limit)


def run_tabu(
    csv_dir,
    time_limit=900,
    pool_lines=60,
    pool_storage=16,
    pool_method="score",
    max_lines=22,
    max_storage=12,
    seed=1,
    solver_preference="gurobi",
    eval_time_limit=25,
    lite_scenarios=None,
    tabu_tenure=8,
    sample_neighbors=6,
):
    rng = random.Random(seed)
    solver_name = pick_solver(preferred=solver_preference)
    eval_model, eval_data = build_model(csv_dir, complementarity=False)
    full_model, full_data = eval_model, eval_data
    pool = build_candidate_pool(full_data, pool_lines, pool_storage, method=pool_method)
    print("Tabu 评估: 全部代表日，运行层为连续松弛（固定投资后为 LP）")

    solver = make_eval_solver(eval_model, solver_name, time_limit=eval_time_limit, mip_rel_gap=0.01)
    cache = {}
    t_end = time.perf_counter() + float(time_limit)
    print(
        f"Tabu: pool={len(pool['lines'])}L/{len(pool['storage'])}H ({pool_method}), "
        f"cap={max_lines}L/{max_storage}H, tenure={tabu_tenure}, time_limit={time_limit}s"
    )

    seed_l, seed_s = make_initial_solution(pool, eval_data, max_lines=min(19, max_lines), max_storage=min(10, max_storage))
    current = _eval(eval_model, eval_data, solver, seed_l, seed_s, cache, eval_time_limit)
    best = current
    tabu = {}  # move_key -> expire_iter
    it = 0
    print(f"  iter=0 best={best.obj:,.2f} lines={len(best.lines)} stor={len(best.storage)}")

    def all_moves(sol):
        moves = []
        for l in pool["lines"]:
            moves.append(("L", l))
        for h in pool["storage"]:
            moves.append(("H", h))
        rng.shuffle(moves)
        return moves

    while time.perf_counter() < t_end:
        it += 1
        candidates = []
        for kind, idx in all_moves(current)[: max(sample_neighbors * 4, 12)]:
            if len(candidates) >= sample_neighbors:
                break
            nl, ns = set(current.lines), set(current.storage)
            if kind == "L":
                if idx in nl:
                    nl.remove(idx)
                else:
                    nl.add(idx)
            else:
                if idx in ns:
                    ns.remove(idx)
                else:
                    ns.add(idx)
            nl, ns = _repair(nl, ns, pool, eval_data, max_lines, max_storage)
            if nl == current.lines and ns == current.storage:
                continue
            sol = _eval(eval_model, eval_data, solver, nl, ns, cache, eval_time_limit)
            move_key = (kind, idx, idx in (current.lines if kind == "L" else current.storage))
            candidates.append((sol, move_key))
            if time.perf_counter() >= t_end:
                break
        if not candidates:
            l, s = _random_individual(pool, eval_data, rng, max_lines, max_storage)
            current = _eval(eval_model, eval_data, solver, l, s, cache, eval_time_limit)
            continue

        def allowed(item):
            sol, mk = item
            expire = tabu.get(mk, -1)
            if sol.obj < best.obj - 1e-6:
                return True  # 渴望准则
            return expire < it

        feasible = [c for c in candidates if allowed(c)] or candidates
        feasible.sort(key=lambda x: x[0].obj)
        current, mk = feasible[0]
        tabu[mk] = it + tabu_tenure
        if current.obj < best.obj - 1e-6:
            best = current
        if it % 3 == 0:
            print(
                f"  iter={it} best={best.obj:,.2f} current={current.obj:,.2f} "
                f"cache={len(cache)} remain={t_end - time.perf_counter():.0f}s"
            )

    return _finalize(full_model, full_data, solver_name, best, cache, it, "Tabu", eval_time_limit)


def _finalize(full_model, full_data, solver_name, best, cache, iters, name, eval_time_limit):
    full_solver = configure_solver(solver_name, time_limit=max(180, eval_time_limit), mip_rel_gap=0.001)
    full_solver.options["OutputFlag"] = 0
    full_cache = {}
    best_full = evaluate_invest(
        full_model,
        full_data,
        full_solver,
        best.lines,
        best.storage,
        full_cache,
        time_limit=max(180, eval_time_limit),
    )
    print(f"\n=== {name} 全代表日复评 ===")
    print(f"搜索迭代/代数: {iters}")
    print(f"目标函数值: {best_full.obj:,.2f}" if math.isfinite(best_full.obj) else "目标函数值: N/A")
    print(f"已选线路: {len(best_full.lines)}")
    print(f"已选储能: {len(best_full.storage)}")
    if best_full.metrics:
        print(f"储能容量: {best_full.metrics.get('total_storage', 0):,.2f} MWh")
        print(f"EENS: {best_full.metrics.get('shed_total', 0):,.2f} MWh/year")
        print(f"弃电量: {best_full.metrics.get('curt_total', 0):,.2f} MWh/year")
    return {
        "name": name,
        "iterations": iters,
        "best_proxy": best,
        "best_full": best_full,
        "cache_size": len(cache),
        "full_model": full_model,
        "full_data": full_data,
        "solver_name": solver_name,
    }


def dump_result(out_path, payload):
    best = payload["best_full"]
    data = {
        "name": payload["name"],
        "iterations": payload["iterations"],
        "obj": best.obj if math.isfinite(best.obj) else None,
        "n_lines": len(best.lines),
        "n_storage": len(best.storage),
        "lines": [list(x) for x in sorted(best.lines)],
        "storage": list(best.storage),
        "metrics": best.metrics,
        "cache_size": payload["cache_size"],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_path}")
    return data


def run_ga_tabu_pipeline(
    csv_dir,
    methods=("ga", "tabu"),
    search_time=900,
    eval_time_limit=25,
    out_dir="results/case300_metaheuristics",
    **kwargs,
):
    lite = kwargs.pop("lite_scenarios", None)
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for m in methods:
        print("\n" + "=" * 60)
        if m == "ga":
            payload = run_ga(
                csv_dir,
                time_limit=search_time,
                eval_time_limit=eval_time_limit,
                lite_scenarios=lite,
                **kwargs,
            )
        elif m == "tabu":
            payload = run_tabu(
                csv_dir,
                time_limit=search_time,
                eval_time_limit=eval_time_limit,
                lite_scenarios=lite,
                **kwargs,
            )
        else:
            raise ValueError(m)
        path = os.path.join(out_dir, f"{payload['name'].lower()}.json")
        summary.append(dump_result(path, payload))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
