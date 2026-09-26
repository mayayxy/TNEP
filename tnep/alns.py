"""Adaptive Large Neighborhood Search for TNEP investment decisions.

ALNS searches binary line/storage investments on a scored candidate pool.
Each neighbor is evaluated by fixing investments and solving the remaining
(continuous) operational model. Elite solutions are re-evaluated on the
full representative-day model and can warm-start exact branch-and-cut.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

from pyomo.environ import SolverFactory, value

from tnep.heuristics import select_heuristic_candidates
from tnep.model import build_model, shrink_tnep_data
from tnep.solver import collect_solution_metrics, configure_solver, pick_solver


@dataclass
class InvestSolution:
    lines: frozenset
    storage: frozenset
    obj: float = math.inf
    metrics: dict = field(default_factory=dict)

    def key(self):
        return (self.lines, self.storage)


def _line_cost(data, l):
    return float(data["c_line"][l])


def _storage_fixed_cost(data, h):
    return float(data["storage_meta"][h]["c_fix"])


def _storage_cap_cost(data, h):
    return float(data["c_cap"]) * float(data["storage_meta"][h]["Ebar"])


def invest_cost_min(data, lines, storage):
    """投资下界：线路年化成本 + 储能场地固定成本（容量可取 0）。"""
    c = sum(_line_cost(data, l) for l in lines)
    c += sum(_storage_fixed_cost(data, h) for h in storage)
    return c


def invest_cost_upper(data, lines, storage):
    """Upper bound on annualized investment if all selected storage are full-sized."""
    c = sum(_line_cost(data, l) for l in lines)
    c += sum(_storage_fixed_cost(data, h) + _storage_cap_cost(data, h) for h in storage)
    return c


def build_candidate_pool(data, pool_lines=40, pool_storage=12, method="score"):
    lines, storage = select_heuristic_candidates(
        data,
        max_lines=pool_lines,
        max_storage=pool_storage,
        method=method,
    )
    line_scores = {l: i for i, l in enumerate(lines)}
    # denser score map from full ranking
    ranked_lines, ranked_storage = select_heuristic_candidates(
        data,
        max_lines=len(data["candidate_lines"]),
        max_storage=len(data["storage_sites"]),
        method="score",
    )
    line_pref = {l: 1.0 / (1 + i) for i, l in enumerate(ranked_lines)}
    stor_pref = {h: 1.0 / (1 + i) for i, h in enumerate(ranked_storage)}
    return {
        "lines": list(lines),
        "storage": list(storage),
        "line_pref": line_pref,
        "stor_pref": stor_pref,
        "line_scores_rank": line_scores,
    }


def _unfix_invest(model):
    for l in model.LC:
        if model.y[l].fixed:
            model.y[l].fixed = False
        model.y[l].set_value(None)
    for h in model.H:
        if model.z[h].fixed:
            model.z[h].fixed = False
        model.z[h].set_value(None)
        if model.E[h].fixed:
            model.E[h].fixed = False
        model.E[h].set_value(None)


def _sync_persistent_invest(solver, model):
    updater = getattr(solver, "update_var", None)
    if updater is None:
        return
    for l in model.LC:
        updater(model.y[l])
    for h in model.H:
        updater(model.z[h])
        updater(model.E[h])


def make_eval_solver(model, solver_name, time_limit=30, mip_rel_gap=0.01):
    """优先 Gurobi persistent，避免每次重建模型。"""
    try:
        persistent = SolverFactory("gurobi_persistent")
        persistent.set_instance(model)
        persistent.options["TimeLimit"] = float(time_limit)
        persistent.options["OutputFlag"] = 0
        persistent.options["MIPGap"] = mip_rel_gap
        print("评估求解器: gurobi_persistent")
        return persistent
    except Exception as exc:
        print(f"persistent 不可用，回退普通 Gurobi ({exc})")
        solver = configure_solver(solver_name, time_limit=time_limit, mip_rel_gap=mip_rel_gap)
        solver.options["OutputFlag"] = 0
        return solver


def apply_invest_fixing(model, lines, storage):
    ls, ss = set(lines), set(storage)
    for l in model.LC:
        model.y[l].fix(1 if l in ls else 0)
    for h in model.H:
        model.z[h].fix(1 if h in ss else 0)
        if h not in ss:
            model.E[h].fix(0)
        elif model.E[h].fixed:
            model.E[h].fixed = False


def evaluate_invest(
    model,
    data,
    solver,
    lines,
    storage,
    cache,
    time_limit=90,
):
    key = (frozenset(lines), frozenset(storage))
    if key in cache:
        return cache[key]

    if invest_cost_min(data, lines, storage) > float(data["Gamma"]) + 1e-6:
        sol = InvestSolution(lines=key[0], storage=key[1], obj=math.inf)
        cache[key] = sol
        return sol

    _unfix_invest(model)
    apply_invest_fixing(model, lines, storage)
    _sync_persistent_invest(solver, model)
    solver.options["TimeLimit"] = float(time_limit)
    solver.options["OutputFlag"] = 0
    try:
        if hasattr(solver, "set_instance"):
            result = solver.solve(save_results=False)
        else:
            result = solver.solve(model, tee=False, load_solutions=False)
    except Exception:
        sol = InvestSolution(lines=key[0], storage=key[1], obj=math.inf)
        cache[key] = sol
        return sol

    try:
        nsol = len(result.solution) if result is not None and hasattr(result, "solution") else 0
        if nsol > 0:
            model.solutions.load_from(result)
        metrics = collect_solution_metrics(model)
        obj = float(metrics["obj"])
        if not math.isfinite(obj):
            raise ValueError("nonfinite obj")
    except Exception:
        sol = InvestSolution(lines=key[0], storage=key[1], obj=math.inf)
        cache[key] = sol
        return sol

    cache[key] = InvestSolution(lines=key[0], storage=key[1], obj=obj, metrics=metrics)
    return cache[key]


# ---- destroy / repair operators ----

def destroy_random_lines(sol, pool, rng, strength=0.3):
    lines = list(sol.lines)
    if not lines:
        return set(), set(sol.storage)
    k = max(1, int(math.ceil(len(lines) * strength)))
    remove = set(rng.sample(lines, min(k, len(lines))))
    return set(sol.lines) - remove, set(sol.storage)


def destroy_random_storage(sol, pool, rng, strength=0.5):
    stor = list(sol.storage)
    if not stor:
        return set(sol.lines), set()
    k = max(1, int(math.ceil(len(stor) * strength)))
    remove = set(rng.sample(stor, min(k, len(stor))))
    return set(sol.lines), set(sol.storage) - remove


def destroy_expensive(sol, pool, data, rng, strength=0.3):
    lines = sorted(sol.lines, key=lambda l: _line_cost(data, l), reverse=True)
    stor = sorted(sol.storage, key=lambda h: _storage_fixed_cost(data, h) + _storage_cap_cost(data, h), reverse=True)
    kl = max(1, int(math.ceil(len(lines) * strength))) if lines else 0
    ks = max(1, int(math.ceil(len(stor) * strength))) if stor else 0
    return set(lines[kl:]), set(stor[ks:])


def destroy_low_pref(sol, pool, rng, strength=0.3):
    lines = sorted(sol.lines, key=lambda l: pool["line_pref"].get(l, 0.0))
    stor = sorted(sol.storage, key=lambda h: pool["stor_pref"].get(h, 0.0))
    kl = max(1, int(math.ceil(len(lines) * strength))) if lines else 0
    ks = max(1, int(math.ceil(len(stor) * strength))) if stor else 0
    return set(lines[kl:]), set(stor[ks:])


def repair_greedy(partial_lines, partial_storage, pool, data, rng, max_lines=15, max_storage=6):
    lines = set(partial_lines)
    storage = set(partial_storage)
    budget = float(data["Gamma"])
    used = invest_cost_upper(data, lines, storage)

    cand_lines = [l for l in pool["lines"] if l not in lines]
    cand_lines.sort(key=lambda l: pool["line_pref"].get(l, 0.0), reverse=True)
    for l in cand_lines:
        if len(lines) >= max_lines:
            break
        c = _line_cost(data, l)
        if used + c <= budget:
            lines.add(l)
            used += c

    cand_h = [h for h in pool["storage"] if h not in storage]
    cand_h.sort(key=lambda h: pool["stor_pref"].get(h, 0.0), reverse=True)
    for h in cand_h:
        if len(storage) >= max_storage:
            break
        c = _storage_fixed_cost(data, h) + _storage_cap_cost(data, h)
        if used + c <= budget:
            storage.add(h)
            used += c
    return lines, storage


def repair_random_greedy(partial_lines, partial_storage, pool, data, rng, max_lines=15, max_storage=6):
    lines = set(partial_lines)
    storage = set(partial_storage)
    budget = float(data["Gamma"])
    used = invest_cost_upper(data, lines, storage)

    cand_lines = [l for l in pool["lines"] if l not in lines]
    rng.shuffle(cand_lines)
    # biased shuffle: keep top half more likely via roulette
    weights = [pool["line_pref"].get(l, 1e-6) for l in cand_lines]
    while cand_lines and len(lines) < max_lines:
        wsum = sum(weights) or 1.0
        r = rng.random() * wsum
        acc = 0.0
        idx = 0
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                idx = i
                break
        l = cand_lines.pop(idx)
        weights.pop(idx)
        c = _line_cost(data, l)
        if used + c <= budget:
            lines.add(l)
            used += c

    cand_h = [h for h in pool["storage"] if h not in storage]
    weights = [pool["stor_pref"].get(h, 1e-6) for h in cand_h]
    while cand_h and len(storage) < max_storage:
        wsum = sum(weights) or 1.0
        r = rng.random() * wsum
        acc = 0.0
        idx = 0
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                idx = i
                break
        h = cand_h.pop(idx)
        weights.pop(idx)
        c = _storage_fixed_cost(data, h) + _storage_cap_cost(data, h)
        if used + c <= budget:
            storage.add(h)
            used += c
    return lines, storage


def repair_budget_fill(partial_lines, partial_storage, pool, data, rng, max_lines=20, max_storage=8):
    """Fill remaining budget with cheapest high-pref assets."""
    lines = set(partial_lines)
    storage = set(partial_storage)
    budget = float(data["Gamma"])
    used = invest_cost_upper(data, lines, storage)

    items = []
    for l in pool["lines"]:
        if l not in lines:
            items.append(("L", l, _line_cost(data, l), pool["line_pref"].get(l, 0.0)))
    for h in pool["storage"]:
        if h not in storage:
            c = _storage_fixed_cost(data, h) + _storage_cap_cost(data, h)
            items.append(("H", h, c, pool["stor_pref"].get(h, 0.0)))
    items.sort(key=lambda x: (x[2] / max(x[3], 1e-9)))
    for kind, idx, c, _ in items:
        if used + c > budget:
            continue
        if kind == "L" and len(lines) < max_lines:
            lines.add(idx)
            used += c
        elif kind == "H" and len(storage) < max_storage:
            storage.add(idx)
            used += c
    return lines, storage


def _roulette(weights, rng):
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc >= r:
            return i
    return len(weights) - 1


def make_initial_solution(pool, data, max_lines=10, max_storage=4):
    lines = set(pool["lines"][:max_lines])
    storage = set(pool["storage"][:max_storage])
    # trim to budget
    while invest_cost_min(data, lines, storage) > float(data["Gamma"]) and lines:
        worst = max(lines, key=lambda l: _line_cost(data, l))
        lines.remove(worst)
    while invest_cost_min(data, lines, storage) > float(data["Gamma"]) and storage:
        worst = max(storage, key=lambda h: _storage_fixed_cost(data, h) + _storage_cap_cost(data, h))
        storage.remove(worst)
    return lines, storage


def run_alns(
    csv_dir,
    time_limit=1800,
    pool_lines=40,
    pool_storage=12,
    pool_method="score",
    max_lines=15,
    max_storage=6,
    seed=1,
    solver_preference="gurobi",
    eval_time_limit=90,
    lite_scenarios=None,
    segment_size=5,
    reaction=0.7,
    start_temp_ratio=0.02,
    cooling=0.995,
    print_header=True,
):
    """Run ALNS and return best investment plan evaluated on the full model."""
    rng = random.Random(seed)
    solver_name = pick_solver(preferred=solver_preference)

    if print_header:
        print(f"数据目录: {csv_dir}")
        print(f"求解器: {solver_name}")
        print(
            f"ALNS: pool={pool_lines}L/{pool_storage}H ({pool_method}), "
            f"cap={max_lines}L/{max_storage}H, time_limit={time_limit}s"
        )

    # Full model for elite re-evaluation / warm-start export
    full_model, full_data = build_model(csv_dir)
    pool = build_candidate_pool(full_data, pool_lines, pool_storage, method=pool_method)

    # Fast evaluation model (optional lite scenarios)
    if lite_scenarios:
        def _transform(d):
            return shrink_tnep_data(d, max_candidate_lines=10**9, max_storage_sites=10**9, keep_scenarios=lite_scenarios)

        eval_model, eval_data = build_model(csv_dir, data_transform=_transform)
        if print_header:
            print(f"ALNS 评估场景(精简): {lite_scenarios}")
    else:
        eval_model, eval_data = full_model, full_data
        if print_header:
            print("ALNS 评估场景: 全代表日")

    solver = configure_solver(solver_name, time_limit=eval_time_limit, mip_rel_gap=0.001)
    solver.options["OutputFlag"] = 0

    cache = {}
    t_end = time.perf_counter() + float(time_limit)

    init_lines, init_storage = make_initial_solution(pool, eval_data, max_lines=min(10, max_lines), max_storage=min(4, max_storage))
    current = evaluate_invest(eval_model, eval_data, solver, init_lines, init_storage, cache, time_limit=eval_time_limit)
    best = current
    if print_header:
        print(f"初始解 obj={current.obj:,.2f} lines={len(current.lines)} stor={len(current.storage)}")

    destroy_ops = ["rnd_line", "rnd_stor", "expensive", "low_pref"]
    repair_ops = ["greedy", "rnd_greedy", "budget_fill"]
    w_d = [1.0] * len(destroy_ops)
    w_r = [1.0] * len(repair_ops)
    score_d = [0.0] * len(destroy_ops)
    score_r = [0.0] * len(repair_ops)
    use_d = [0] * len(destroy_ops)
    use_r = [0] * len(repair_ops)

    temp0 = abs(current.obj) * start_temp_ratio if math.isfinite(current.obj) else 1e9
    temp = temp0
    it = 0
    history = []

    while time.perf_counter() < t_end:
        it += 1
        di = _roulette(w_d, rng)
        ri = _roulette(w_r, rng)
        dname, rname = destroy_ops[di], repair_ops[ri]
        use_d[di] += 1
        use_r[ri] += 1

        strength = rng.uniform(0.2, 0.5)
        if dname == "rnd_line":
            pl, ps = destroy_random_lines(current, pool, rng, strength)
        elif dname == "rnd_stor":
            pl, ps = destroy_random_storage(current, pool, rng, strength)
        elif dname == "expensive":
            pl, ps = destroy_expensive(current, pool, eval_data, rng, strength)
        else:
            pl, ps = destroy_low_pref(current, pool, rng, strength)

        if rname == "greedy":
            nl, ns = repair_greedy(pl, ps, pool, eval_data, rng, max_lines, max_storage)
        elif rname == "rnd_greedy":
            nl, ns = repair_random_greedy(pl, ps, pool, eval_data, rng, max_lines, max_storage)
        else:
            nl, ns = repair_budget_fill(pl, ps, pool, eval_data, rng, max_lines, max_storage)

        cand = evaluate_invest(eval_model, eval_data, solver, nl, ns, cache, time_limit=eval_time_limit)
        if not math.isfinite(cand.obj):
            score_d[di] += 0.1
            score_r[ri] += 0.1
        else:
            delta = cand.obj - current.obj
            accept = False
            reward = 0.0
            if cand.obj < best.obj - 1e-6:
                best = cand
                current = cand
                accept = True
                reward = 3.0
            elif cand.obj < current.obj - 1e-6:
                current = cand
                accept = True
                reward = 2.0
            elif temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-9)):
                current = cand
                accept = True
                reward = 1.0
            if accept:
                score_d[di] += reward
                score_r[ri] += reward

        temp *= cooling
        history.append({"iter": it, "best": best.obj, "current": current.obj, "destroy": dname, "repair": rname})

        if it % segment_size == 0:
            for i in range(len(w_d)):
                if use_d[i] > 0:
                    w_d[i] = reaction * (score_d[i] / use_d[i]) + (1 - reaction) * w_d[i]
                score_d[i] = 0.0
                use_d[i] = 0
            for i in range(len(w_r)):
                if use_r[i] > 0:
                    w_r[i] = reaction * (score_r[i] / use_r[i]) + (1 - reaction) * w_r[i]
                score_r[i] = 0.0
                use_r[i] = 0
            if print_header:
                print(
                    f"  iter={it} best={best.obj:,.2f} current={current.obj:,.2f} "
                    f"cache={len(cache)} temp={temp:.3e}"
                )

    # Re-evaluate best pattern on full representative-day model
    full_solver = configure_solver(solver_name, time_limit=max(120, eval_time_limit), mip_rel_gap=0.001)
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

    return {
        "best_proxy": best,
        "best_full": best_full,
        "pool": pool,
        "iterations": it,
        "history": history,
        "full_model": full_model,
        "full_data": full_data,
        "solver_name": solver_name,
    }


def warmstart_exact_from_invest(
    csv_dir,
    lines,
    storage,
    time_limit=3600,
    mip_rel_gap=0.001,
    solver_preference="gurobi",
    gurobi_seed=None,
):
    """Solve full MILP with MIP start from a given investment pattern."""
    from tnep.solver import extract_bounds_and_gap

    solver_name = pick_solver(preferred=solver_preference)
    model, data = build_model(csv_dir)
    ls, ss = set(lines), set(storage)

    # MIP start values (unfixed)
    for l in model.LC:
        model.y[l].set_value(1.0 if l in ls else 0.0)
    for h in model.H:
        model.z[h].set_value(1.0 if h in ss else 0.0)
        model.E[h].set_value(float(data["storage_meta"][h]["Ebar"]) if h in ss else 0.0)

    solver = configure_solver(
        solver_name,
        time_limit=time_limit,
        mip_rel_gap=mip_rel_gap,
        gurobi_seed=gurobi_seed,
    )
    solver.options["OutputFlag"] = 1
    t0 = time.perf_counter()
    result = solver.solve(model, tee=False, warmstart=True, load_solutions=False)
    solve_time = time.perf_counter() - t0
    lb, ub, gap = extract_bounds_and_gap(result)

    out = {
        "termination_condition": str(result.solver.termination_condition),
        "solve_time_sec": solve_time,
        "lb": lb,
        "ub": ub,
        "mip_gap": gap,
        "ok": True,
        "solution_source": "alns_warmstart_exact",
        "tnep_data": data,
    }
    if len(result.solution) > 0:
        model.solutions.load_from(result)
        out.update(collect_solution_metrics(model))
    else:
        out.update(
            {
                "obj": None,
                "selected_lines": None,
                "selected_storage": None,
                "total_storage": None,
                "invest_line": None,
                "invest_stor": None,
                "shed_total": None,
                "curt_total": None,
                "op_cost": None,
                "gamma": None,
            }
        )
    return out


def solve_alns_pipeline(
    csv_dir,
    alns_time_limit=1200,
    exact_time_limit=3600,
    mip_rel_gap=0.001,
    pool_lines=40,
    pool_storage=12,
    pool_method="score",
    max_lines=15,
    max_storage=6,
    seed=1,
    solver_preference="gurobi",
    then_exact=True,
    lite_scenarios=None,
    eval_time_limit=90,
):
    if lite_scenarios is None:
        lite_scenarios = ["S1", "S6", "S9", "S12"]

    alns = run_alns(
        csv_dir,
        time_limit=alns_time_limit,
        pool_lines=pool_lines,
        pool_storage=pool_storage,
        pool_method=pool_method,
        max_lines=max_lines,
        max_storage=max_storage,
        seed=seed,
        solver_preference=solver_preference,
        lite_scenarios=lite_scenarios,
        eval_time_limit=eval_time_limit,
    )
    best = alns["best_full"]
    print("\n=== ALNS 结果（全代表日复评）===")
    print(f"迭代次数: {alns['iterations']}")
    print(f"目标函数值: {best.obj:,.2f}" if math.isfinite(best.obj) else "目标函数值: N/A")
    print(f"已选线路: {len(best.lines)}")
    print(f"已选储能: {len(best.storage)}")
    if best.metrics:
        print(f"储能容量: {best.metrics.get('total_storage', 0):,.2f} MWh")
        print(f"EENS: {best.metrics.get('shed_total', 0):,.2f} MWh/year")
        print(f"弃电量: {best.metrics.get('curt_total', 0):,.2f} MWh/year")

    exact = None
    if then_exact and math.isfinite(best.obj):
        print("\n=== ALNS 热启动精确求解 ===")
        exact = warmstart_exact_from_invest(
            csv_dir,
            best.lines,
            best.storage,
            time_limit=exact_time_limit,
            mip_rel_gap=mip_rel_gap,
            solver_preference=solver_preference,
        )
        from tnep.solver import print_result_block

        print_result_block("精确求解(ALNS warm-start)", exact)
    return alns, exact
