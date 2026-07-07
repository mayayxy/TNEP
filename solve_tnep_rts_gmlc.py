"""RTS-GMLC 数据集的 TNEP + 储能联合规划求解入口。"""

import argparse
import os

from solve_tnep_case57 import solve_once


DEFAULT_CSV_DIR = os.path.join(os.path.dirname(__file__), "dataset_csv_rts_gmlc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default=DEFAULT_CSV_DIR, help="数据目录，默认 dataset_csv_rts_gmlc")
    parser.add_argument("--time-limit", type=int, default=120, help="求解时间上限（秒）")
    parser.add_argument("--heuristic", action="store_true", help="启发式筛选投资候选后再求解")
    parser.add_argument("--heuristic-lines", type=int, default=10, help="启发式保留的候选线路数量")
    parser.add_argument("--heuristic-storage", type=int, default=4, help="启发式保留的储能候选数量")
    args = parser.parse_args()

    solve_once(
        args.csv_dir,
        time_limit=args.time_limit,
        heuristic=args.heuristic,
        heuristic_lines=args.heuristic_lines,
        heuristic_storage=args.heuristic_storage,
    )
