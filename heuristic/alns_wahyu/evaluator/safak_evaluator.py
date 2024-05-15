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
    def __init__(self, insertion_mode:str="wall-building", cargo_sort_criterion="random") -> None:
        """
            , cargo_sort_criterion:str="random"
            cargo_sort_criterion=["random", "lwh,dec", "lwh,inc", "h-base-area,dec-inc",  "base-area,dec","base-area,inc", "wall-area,dec", "wall-area,inc", "column-area,dec", "column-area,inc"]

        """
        super().__init__()
        self.insertion_mode=insertion_mode
        self.cargo_sort_criterion=cargo_sort_criterion
    
    def init_cargo_priority(self, solution:Solution)->Solution:
        if self.cargo_sort_criterion == "random":
            return solution
        sort_criterion, order = self.cargo_sort_criterion.split(sep=",")
        cargo_dims = solution.cargo_dims
        if sort_criterion=="lwh":
            sorted_idx = np.lexsort((cargo_dims[:, 2], cargo_dims[:, 1], cargo_dims[:, 0]))
            priority = np.argsort(sorted_idx)
        elif sort_criterion=="wall-area":
            wall_area = cargo_dims[:,1]*cargo_dims[:,2]
            priority = wall_area
        elif sort_criterion=="base-area":
            base_area = cargo_dims[:,0]*cargo_dims[:,1]
            priority = base_area
        elif sort_criterion == "column-area":
            column_area = cargo_dims[:,0]*cargo_dims[:,2]
            priority = column_area
        elif sort_criterion=="h-base-area":
            height = cargo_dims[:,2]
            base_area = cargo_dims[:,0]*cargo_dims[:,1]
            order1, order2 = order.split("-")
            if order1 == "dec":
                height = -height
            if order2 == "dec":
                base_area = -base_area
            sorted_idx = np.lexsort((base_area, height))
            priority = np.argsort(sorted_idx)
        if order == "dec":
            priority = -priority
        solution.cargo_type_priority = priority
        return  solution

    def solve(self, 
              ccm:np.ndarray, 
              df_cargos: pd.DataFrame, 
              df_container: pd.DataFrame)->Solution:
        cargo_type_list = create_cargo_type_list(df_cargos, ccm)
        container_type_list = create_container_type_list(df_container)
        print(df_container)
        prob = Problem(cargo_type_list, container_type_list)
        solution = Solution(prob)
        solution = self.init_cargo_priority(solution)
        solution = add_container(solution, 0)
        solution = constructive_heuristic(solution, insertion_mode=self.insertion_mode)
        return solution