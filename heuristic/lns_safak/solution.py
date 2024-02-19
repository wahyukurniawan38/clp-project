
import numpy as np

from heuristic.lns_safak.utils import init_cargo_priority, init_rotation_priority
from solver.problem import Problem
from solver.solution import SolutionBase


class Solution(SolutionBase):
    def __init__(self, problem: Problem, **kwargs):
        super().__init__(problem, **kwargs)
        
        self.cargo_priority = init_cargo_priority(problem, kwargs.get("cargo_priority"))
        self.rotation_priority = init_rotation_priority(problem, kwargs.get("rotation_priority"))
        self.is_feasible = kwargs.get("is_feasible")
        if self.is_feasible is None:
            self.is_feasible = False