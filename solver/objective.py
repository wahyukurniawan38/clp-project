from typing import Tuple

import numpy as np

from solver.solution import SolutionBase

"""
    out:
        total_packed_volume (packed_volume/all_cargo_volume)
        total_profit ((packed_cargo_profit-container_cost)/all_cargo_profit)
"""
def compute_objective(solution: SolutionBase)-> Tuple[float,float]:
    is_container_used = solution.container_filled_volumes>0
    container_cost = np.sum(solution.container_costs[is_container_used])

    is_cargo_packed = solution.cargo_container_maps > -1
    cargo_profit = np.sum(solution.cargo_costs[is_cargo_packed])
    total_cargo_profit = np.sum(solution.cargo_costs)
    cost_obj = (cargo_profit-container_cost)/total_cargo_profit

    packed_volume = np.sum(solution.cargo_volumes[is_cargo_packed])
    total_volume = np.sum(solution.cargo_volumes)
    vol_obj = packed_volume/total_volume
    return [vol_obj, cost_obj]

def is_better(solution_a:SolutionBase, solution_b: SolutionBase):
    vol_obj_a, cos_obj_a = compute_objective(solution_a)
    vol_obj_b, cos_obj_b = compute_objective(solution_b)
    if vol_obj_a > vol_obj_b:
        return True
    if vol_obj_a == vol_obj_b:
        return cos_obj_a > cos_obj_b
    return False