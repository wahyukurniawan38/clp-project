from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

from heuristic.alns_wahyu.evaluator.evaluation_result import EvaluationResult
from solver.solution import SolutionBase
from solver.problem import Problem

class Evaluator(ABC):
    def __init__(self) -> None:
        return

    @abstractmethod
    def solve(self, 
              ccm:np.ndarray, 
              df_cargos: pd.DataFrame, 
              df_container: pd.DataFrame)->SolutionBase:
        pass

    def evaluate(self, 
                 x: np.ndarray, 
                 df_cargos: pd.DataFrame, 
                 df_containers: pd.DataFrame,
                 omega:float=0.99)->EvaluationResult:
        solution_list: List[SolutionBase] = []
        for container_idx, chosen_cargo_mask in enumerate(x):
            if not np.any(chosen_cargo_mask):
                solution_list += [SolutionBase(Problem([],[]))]
                continue
            solution = self.solve(chosen_cargo_mask, df_cargos, df_containers.iloc[[container_idx]])
            solution_list += [solution]
        return EvaluationResult(df_cargos, df_containers, x, solution_list, omega)

            