from copy import deepcopy
from random import randint
from typing import Tuple

import numpy as np
import pandas as pd

from heuristic.alns_wahyu.evaluator.evaluation_result import (EvaluationResult,
                                                              create_copy)
from heuristic.alns_wahyu.evaluator.evaluator import Evaluator
from heuristic.alns_wahyu.operator.utils import *
from solver.solution import SolutionBase


def feasibility_repair(eval_result:EvaluationResult, 
                       evaluator:Evaluator,
                       max_iter:int):
    df_cargos = eval_result.df_cargos
    df_containers = eval_result.df_containters
    omega = eval_result.omega
    for t in range(max_iter):
        # eval_result = repair_cargo_packing_feasibility(eval_result, evaluator)
        # eval_result = repair_cog(eval_result)
        if eval_result.is_feasible:
            break
        # r = randint(0,3)
        r=0
        is_success = False
        if r==0:
            is_success, new_x = move_heaviest_items_to_random_container(eval_result)
        new_eval_result = evaluator.evaluate(new_x, df_cargos, df_containers, omega)
        # if not is_success:
            # swap cargos from two containers

        
    return eval_result


def find_least_loaded_container(eval_result:EvaluationResult)->int:
    item_loads = eval_result.cargo_volumes*eval_result.cargo_weights
    # is_container_filled = 

def find_suitable_random_container(eval_result:EvaluationResult, 
                                   heaviest_infeasible_cargo_idx:np.ndarray)->int:
    hi_volume = np.sum(eval_result.cargo_volumes[heaviest_infeasible_cargo_idx])
    hi_weight = np.sum(eval_result.cargo_weights[heaviest_infeasible_cargo_idx])
    remaining_container_volumes = np.asanyarray([solution.container_max_volumes[0]-solution.container_filled_volumes[0] for solution in eval_result.solution_list])
    remaining_container_weights = np.asanyarray([solution.container_max_weights[0]-solution.container_filled_weights[0] for solution in eval_result.solution_list])
    container_has_enough_weight_cap = hi_weight<=remaining_container_weights
    container_has_enough_vol_cap = hi_volume<=remaining_container_volumes
    is_container_suitable = np.logical_and(container_has_enough_vol_cap, container_has_enough_weight_cap)
    if not np.any(is_container_suitable):
        return -1
    suitable_containers = np.where(is_container_suitable)[0]
    selected_container = np.random.choice(suitable_containers)
    return selected_container


def move_heaviest_items_to_random_container(eval_result:EvaluationResult)->Tuple[bool, np.ndarray]:
    heaviest_infeasible_cargo_idx = find_heaviest_infeasible_cargos(eval_result)
    container_idx = find_suitable_random_container(eval_result, heaviest_infeasible_cargo_idx)
    if container_idx == -1:
        return False, None
    new_x = eval_result.x.copy()
    new_x[:, heaviest_infeasible_cargo_idx] = 0
    new_x[container_idx,heaviest_infeasible_cargo_idx] = 1
    return True, new_x

# i think we can make this recursive
# 1. first we need to find all failed cargos that need to be inserted
# 2. remove those failed cargos from their original containers
# 3. start recursion, 
#     4. try to insert them to containers, starting from the most filled one
#     5. check if successful, else collect remaining failed cargo ids
#     6. remove the most filled container, the one we just try to load into
#     7. go to step 3
# 4. if there is still failed cargo, then just return so. nothing left to do then, i suppose
def repair_cargo_packing_feasibility(eval_result:EvaluationResult,
                                     evaluator: Evaluator)->EvaluationResult:
    eval_result = remove_unpacked_cargo_from_result(eval_result)
    container_utilities = eval_result.container_utilities
    allowed_container_idx = np.argsort(-container_utilities)
    eval_result = insert_unpacked_cargo_to_containers(eval_result, evaluator, allowed_container_idx)
    return eval_result



# now repair the cog by shifting the items
def repair_cog(eval_result: EvaluationResult):
    for s_idx, solution in enumerate(eval_result.solution_list):
        if not solution.is_cog_feasible:
            eval_result.solution_list[s_idx] = repair_solution_cog_by_shifting(solution)
    return eval_result


# if necessary shift to do is less than what's possible,
# then just return the current solution, nothing can be done?
# or keep shifting it?
# for now keep shifting it
def repair_solution_cog_by_shifting(solution: SolutionBase):
    current_cog = solution.container_cogs[0,:]
    cog_tolerance = solution.container_cog_tolerances
    cc_real_dims = solution.real_cargo_dims
    xy_container_dim = solution.container_dims[0,:2]
    xy_corner_points = solution.positions[:,:2] + cc_real_dims[:,:2]
    xy_container_mid = xy_container_dim/2
    necessary_shift = np.clip((xy_container_mid-cog_tolerance)-current_cog, a_min=0, a_max=None)
    max_shift = xy_container_dim-np.max(xy_corner_points,axis=0)
    possible_shift = np.clip(necessary_shift, a_min=None, a_max=max_shift)
    solution.positions[:,:2] += possible_shift[np.newaxis,:]
    solution.container_cogs[0,:] += possible_shift
    return solution