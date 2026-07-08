"""算例注册表与路径配置。"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = {
    "rts24": {
        "label": "IEEE RTS-24",
        "dataset_dir": "dataset_csv",
        "description": "24 母线论文原型，6 代表日 × 4 小时",
    },
    "case57": {
        "label": "IEEE 57-bus",
        "dataset_dir": "dataset_csv_case57",
        "description": "PyPower case57，12 代表日 × 24 小时",
    },
    "case300": {
        "label": "IEEE 300-bus",
        "dataset_dir": "dataset_csv_case300",
        "description": "PyPower case300，12 代表日 × 24 小时",
    },
    "rts_gmlc": {
        "label": "RTS-GMLC",
        "dataset_dir": "dataset_csv_rts_gmlc",
        "description": "NREL RTS-GMLC 真实时序，12 代表日 × 24 小时",
    },
}


def list_cases():
    return list(CASES.keys())


def dataset_dir(case, root=None):
    if case not in CASES:
        known = ", ".join(list_cases())
        raise ValueError(f"未知算例 '{case}'，可选: {known}")
    base = root if root is not None else ROOT
    return os.path.join(base, CASES[case]["dataset_dir"])
