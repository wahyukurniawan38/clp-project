import pathlib

import numpy as np
import pandas as pd

from heuristic.alns_wahyu.evaluator.safak_evaluator import SafakEvaluator
from heuristic.alns_wahyu.utils import create_cargo_type_list, create_container_type_list
from heuristic.lns_safak.solution import Solution
from heuristic.lns_safak.constructive import constructive_heuristic
from heuristic.utils import add_container
from solver.problem import Problem
from solver.utils import visualize_box


def run():
    data_path = pathlib.Path()/"instances"/"data_from_wahyu"/"dummy_data_small.xlsx"
    df_cargos = pd.read_excel(data_path.absolute(),sheet_name='Item', header=0)
    df_containers = pd.read_excel(data_path.absolute(),sheet_name='Container', header=0)
    evaluator = SafakEvaluator()

    while True:
        x0 = np.random.random(size=(len(df_cargos),)).round().astype(int)
        x1 = 1-x0
        x = np.stack([x0,x1])
        eval_result = evaluator.evaluate(x, df_cargos, df_containers)
        print(eval_result.is_all_cargo_packed)
        print(eval_result.is_all_cog_feasible)
        exit()

if __name__ == "__main__":
    run()