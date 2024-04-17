import pathlib

import numpy as np
import pandas as pd

from heuristic.alns_wahyu.evaluator.safak_evaluator import SafakEvaluator
from heuristic.alns_wahyu.initialization import initialize_x
from heuristic.alns_wahyu.objective import compute_obj
from heuristic.lns_safak.solution import Solution
from heuristic.lns_safak.constructive import constructive_heuristic
from heuristic.utils import add_container
from solver.problem import Problem
from solver.utils import visualize_box


def run():
    data_path = pathlib.Path()/"instances"/"data_from_wahyu"/"Data ALNS.xlsx"
    df_cargos = pd.read_excel(data_path.absolute(),sheet_name='Item', header=0)
    df_containers = pd.read_excel(data_path.absolute(),sheet_name='Container', header=0)
    evaluator = SafakEvaluator()
    x = initialize_x(df_cargos, df_containers)
    score = compute_obj(x, df_cargos, df_containers)
    eval_result = evaluator.evaluate(x, df_cargos, df_containers)
    print(eval_result.is_feasible)
    print(eval_result.is_all_cargo_packed, eval_result.is_all_cog_feasible)
    

if __name__ == "__main__":
    run()