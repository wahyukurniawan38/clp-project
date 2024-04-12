import math

import numpy as np
import pandas as pd

from heuristic.alns_wahyu.evaluator.evaluator import Evaluator
from heuristic.alns_wahyu.initialization import initialize_x
from heuristic.alns_wahyu.objective import compute_obj

def solve(df_cargos:pd.DataFrame,
          df_containers:pd.DataFrame,
          destroy_operators,
          repair_operators,
          evaluator: Evaluator,
          max_iteration:int=10,
          omega:float=0.99,
          a:float=0.9,
          b1:float=1.5,
          b2:float=0.6
          ):
    initial_x = initialize_x(df_cargos, df_containers)
    score = compute_obj(initial_x, df_cargos, df_containers, omega)

    destroy_scores = np.ones((len(destroy_operators),))
    repair_scores = np.ones((len(repair_operators),))
    destroy_counts = np.zeros((len(destroy_operators),))
    repair_counts= np.zeros((len(repair_operators),))

    best_x = initial_x.copy()
    best_score = score

    current_x = initial_x.copy()
    current_score = score

    for t in range(max_iteration):
        p_destroy = destroy_scores/destroy_scores.sum()
        p_repair = repair_scores/repair_scores.sum()
        destroy_op_idx = np.random.choice(len(destroy_operators), p=p_destroy)
        destroy_op = destroy_operators[destroy_op_idx]
        destroy_counts[destroy_op_idx]+=1
        repair_op_idx = np.random.choice(len(repair_operators), p=p_repair)
        repair_op = repair_operators[repair_op_idx]
        repair_counts[repair_op_idx]+=1
        next_x = current_x.copy()
        next_x = destroy_op(next_x)
        next_x = repair_op(next_x)
        next_score = compute_obj(next_x, df_cargos, df_containers, omega)
        if next_score > current_score:
            