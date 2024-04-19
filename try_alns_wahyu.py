import pathlib

import numpy as np
import pandas as pd

from heuristic.alns_wahyu.evaluator.safak_evaluator import SafakEvaluator
from heuristic.alns_wahyu.initialization import initialize_x
from heuristic.alns_wahyu.operator.feasibility_repair import feasibility_repair
from solver.utils import visualize_box


def run():
    data_path = pathlib.Path()/"instances"/"data_from_wahyu"/"Data ALNS.xlsx"
    df_cargos = pd.read_excel(data_path.absolute(),sheet_name='Item', header=0)
    df_containers = pd.read_excel(data_path.absolute(),sheet_name='Container', header=0)
    evaluator = SafakEvaluator()
    x = initialize_x(df_cargos, df_containers)
  
    eval_result = evaluator.evaluate(x, df_cargos, df_containers)
    print(eval_result.is_all_cargo_packed, eval_result.is_all_cog_feasible)
    eval_result = feasibility_repair(eval_result, evaluator, max_iter=10)
    print(eval_result.is_all_cargo_packed, eval_result.is_all_cog_feasible)
    for s_idx, solution in enumerate(eval_result.solution_list):
        container_dim = solution.container_dims[0]
        cc_positions = solution.positions
        cc_dims = solution.cargo_dims
        cc_rotation_mats = solution.rotation_mats
        visualize_box(container_dim,
                  cc_positions,
                  cc_dims,
                  cc_rotation_mats)
        
        


if __name__ == "__main__":
    run()