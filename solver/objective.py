from typing import Tuple

import numpy as np

from solver.solution import SolutionBase

"""
    out:
    (maximize) volume packed -> prioritize packing all items
    (maximize) total_profit ((packed_cargo_profit-container_cost)/all_cargo_profit)
    (minimize) utilization_distribution (max(utilization of all used containers)-min(utilization of all used containers))
        note: this has the same formula as load_distribution for single type container
        i.e., max(filled_volume)-min(filled_volume) 

"""
def compute_objective(solution: SolutionBase)-> Tuple[float,float]:
    is_container_used = solution.container_filled_volumes>0
    container_cost = np.sum(solution.container_costs[is_container_used])

    is_cargo_packed = solution.cargo_container_maps > -1
    cargo_profit = np.sum(solution.cargo_costs[is_cargo_packed])
    total_cargo_profit = np.sum(solution.cargo_costs)
    cost_obj = (cargo_profit-container_cost)/total_cargo_profit

    container_utilization = solution.container_filled_volumes[is_container_used]/solution.container_max_volumes[is_container_used]
    max_util = np.max(container_utilization)
    min_util = np.min(container_utilization)
    load_dist_obj = max_util-min_util

    volume_packed = np.sum(solution.cargo_volumes[is_cargo_packed])/np.sum(solution.cargo_volumes)

    return [volume_packed, cost_obj, load_dist_obj]

def is_better(solution_a:SolutionBase, solution_b: SolutionBase):
    volume_packed_a, cost_obj_a, load_dist_obj_a = compute_objective(solution_a)
    volume_packed_b, cost_obj_b, load_dist_obj_b = compute_objective(solution_b)
    scalarized_obj_a = cost_obj_a-load_dist_obj_a
    scalarized_obj_b = cost_obj_b-load_dist_obj_b
    if volume_packed_a > volume_packed_b:
        return True
    if volume_packed_a == volume_packed_b:
        return scalarized_obj_a > scalarized_obj_b
    return False