
import numpy as np

from heuristic.lns_safak.utils import init_cargo_type_priority, init_cargo_type_rotation_sorted_idx, init_default_cargo_type_rotation_sorted_idx
from solver.problem import Problem
from solver.solution import SolutionBase


class Solution(SolutionBase):
    def __init__(self, problem: Problem, **kwargs):
        super().__init__(problem, **kwargs)
        
        self.cargo_type_priority = init_cargo_type_priority(problem, kwargs.get("cargo_type_priority"))
        self.cargo_type_rotation_sorted_idx = init_cargo_type_rotation_sorted_idx(problem, kwargs.get("cargo_type_rotation_sorted_idx"))
        self.default_cargo_type_rotation_sorted_idx = init_default_cargo_type_rotation_sorted_idx(problem, kwargs.get("default_cargo_type_rotation_sorted_idx"))
        
        self.is_feasible = kwargs.get("is_feasible")
        if self.is_feasible is None:
            self.is_feasible = False

def create_copy(solution: Solution)->Solution:
    new_solution = Solution(solution.problem,
                            positions=solution.positions,
                            cargo_container_maps=solution.cargo_container_maps,
                            rotation_mats=solution.rotation_mats,
                            nums_container_used=solution.nums_container_used,
                            container_dims=solution.container_dims,
                            container_max_volumes=solution.container_max_volumes,
                            container_filled_volumes=solution.container_filled_volumes,
                            container_max_weights=solution.container_max_weights,
                            container_filled_weights=solution.container_filled_weights,
                            container_costs=solution.container_costs,
                            container_types=solution.container_types,
                            cargo_type_priority=solution.cargo_type_priority,
                            cargo_type_rotation_sorted_idx=solution.cargo_type_rotation_sorted_idx,
                            default_cargo_type_rotation_sorted_idx=solution.default_cargo_type_rotation_sorted_idx,
                            is_feasible=solution.is_feasible)
    return new_solution