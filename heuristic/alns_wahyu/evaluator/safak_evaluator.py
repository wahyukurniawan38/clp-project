from typing import List, Tuple

import numpy as np
import pandas as pd


from heuristic.alns_wahyu.evaluator.evaluator import Evaluator
from heuristic.alns_wahyu.utils import create_cargo_type_list, create_container_type_list
from heuristic.lns_safak.constructive import constructive_heuristic
from heuristic.lns_safak.solution import Solution
from heuristic.utils import add_container
from solver.problem import Problem

class SafakEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()
    
    def solve(self, 
              ccm:np.ndarray, 
              df_cargos: pd.DataFrame, 
              df_containers: pd.DataFrame)->Solution:
        cargo_type_list = create_cargo_type_list(df_cargos, ccm)
        container_type_list = create_container_type_list(df_containers)[:1]
        prob = Problem(cargo_type_list, container_type_list)
        solution = Solution(prob)
        solution = add_container(solution, 0)
        solution = constructive_heuristic(solution)
        return solution