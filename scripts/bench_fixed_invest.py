import time

from pyomo.environ import value

from tnep.config import dataset_dir
from tnep.heuristics import select_heuristic_candidates
from tnep.model import build_model
from tnep.solver import collect_solution_metrics, configure_solver


def main():
    csv = dataset_dir("case300")
    print("building model...")
    t0 = time.perf_counter()
    model, data = build_model(csv)
    print(f"build {time.perf_counter() - t0:.1f}s")
    lines, stor = select_heuristic_candidates(data, 10, 4, "diversity")
    ls, ss = set(lines), set(stor)
    for l in model.LC:
        model.y[l].fix(1 if l in ls else 0)
    for h in model.H:
        model.z[h].fix(1 if h in ss else 0)
        if h not in ss:
            model.E[h].fix(0)
    solver = configure_solver("gurobi", time_limit=120, mip_rel_gap=0.001)
    solver.options["OutputFlag"] = 0
    t0 = time.perf_counter()
    res = solver.solve(model, tee=False)
    print(
        f"fixed-invest solve {time.perf_counter() - t0:.1f}s "
        f"status={res.solver.termination_condition}"
    )
    model.solutions.load_from(res)
    m = collect_solution_metrics(model)
    print(
        f"obj={m['obj']:,.2f} lines={len(m['selected_lines'])} "
        f"stor={len(m['selected_storage'])} E={m['total_storage']:.0f}"
    )


if __name__ == "__main__":
    main()
