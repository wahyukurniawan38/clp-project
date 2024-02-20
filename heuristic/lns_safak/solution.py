
import numpy as np

from heuristic.lns_safak.utils import init_cargo_type_priority, init_cargo_type_rotation_sorted_idx
from solver.problem import Problem
from solver.solution import SolutionBase


class Solution(SolutionBase):
    def __init__(self, problem: Problem, **kwargs):
        super().__init__(problem, **kwargs)
        
        self.cargo_type_priority = init_cargo_type_priority(problem, kwargs.get("cargo_type_priority"))
        self.cargo_type_rotation_sorted_idx = init_cargo_type_rotation_sorted_idx(problem, kwargs.get("cargo_type_rotation_sorted_idx"))
        self.is_feasible = kwargs.get("is_feasible")
        if self.is_feasible is None:
            self.is_feasible = False