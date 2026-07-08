"""数据集构建注册表。"""

from tnep.config import CASES, dataset_dir


def _load_builder(case):
    if case == "rts24":
        from tnep.datasets.rts24 import build

        return build
    if case == "case57":
        from tnep.datasets.case57 import build

        return build
    if case == "case300":
        from tnep.datasets.case300 import build

        return build
    if case == "rts_gmlc":
        from tnep.datasets.rts_gmlc import build

        return build
    known = "rts24, case57, case300, rts_gmlc"
    raise ValueError(f"未知算例 '{case}'，可选: {known}")


def list_builders():
    return ["rts24", "case57", "case300", "rts_gmlc"]


def build_dataset(case, out_dir=None):
    build = _load_builder(case)
    target = out_dir if out_dir is not None else dataset_dir(case)
    print(f"构建算例: {case} ({CASES[case]['label']})")
    print(f"输出目录: {target}")
    build(target)
