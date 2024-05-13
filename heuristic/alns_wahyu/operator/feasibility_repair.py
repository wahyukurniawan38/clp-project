from copy import deepcopy
from random import randint
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from heuristic.alns_wahyu.evaluator.evaluation_result import EvaluationResult, create_copy
from heuristic.alns_wahyu.evaluator.evaluator import Evaluator
from heuristic.alns_wahyu.operator.utils import *
from solver.solution import SolutionBase


def feasibility_repair(eval_result:EvaluationResult, 
                       evaluator:Evaluator): #,max_iter:int
    df_cargos = eval_result.df_cargos
    df_containers = eval_result.df_containters
    omega = eval_result.omega
    while True:#for t in range(max_iter):
        eval_result = repair_cargo_packing_feasibility(eval_result, evaluator)
        eval_result = repair_cog(eval_result)
        if eval_result.is_feasible:
           break
        r = randint(0,3)
        is_success = False
        new_x = None
        if r==0:
            is_success, new_x = move_heaviest_cargo_to_random_container(eval_result)
        elif r==1:
            is_success, new_x = move_heaviest_cargo_to_least_loaded_container(eval_result)
        elif r==2:
            is_success, new_x = move_heaviest_cargo_to_second_least_loaded_container(eval_result)
        else:
            is_success, new_x = move_partial_cargo_to_least_loaded_container(eval_result)
        if not is_success:
            # swap cargos from two containers
            least_lead_container_idx = find_least_loaded_container(eval_result)
            most_load_container_idx = find_most_loaded_container(eval_result)
            ct_heavy_idx = most_load_container_idx
            ct_light_idx = least_lead_container_idx
            if most_load_container_idx == least_lead_container_idx:
                second_least_load_container_idx = find_second_least_loaded_container(eval_result)
                ct_light_idx = second_least_load_container_idx
            if ct_heavy_idx != ct_light_idx:
                new_x = constrained_swap_items(eval_result, ct_heavy_idx, ct_light_idx)
        if new_x is not None:
            new_eval_result = evaluator.evaluate(new_x, df_cargos, df_containers, omega)
            eval_result = new_eval_result
    return eval_result

def constrained_swap_items(eval_result:EvaluationResult, ct_heavy_idx:int, ct_light_idx:int)->np.ndarray:
    new_x = eval_result.x.copy()
    cargo_loads = eval_result.cargo_loads
    c_idx_in_heavy_ct = np.where(new_x[ct_heavy_idx,:])[0]
    c_idx_in_light_ct = np.where(new_x[ct_light_idx,:])[0]
    cargo_load_in_heavy_ct = cargo_loads[c_idx_in_heavy_ct]
    c_heavy_sorted_idx = c_idx_in_heavy_ct[np.argsort(-cargo_load_in_heavy_ct)]
    # we shuffle the index in the light container,
    # so that it does not become too deterministic
    # e.g., we swap the same cargo on every iteration
    np.random.shuffle(c_idx_in_light_ct)
    for c_h_idx in c_heavy_sorted_idx:
        for c_l_idx in c_idx_in_light_ct:
            if can_swap(eval_result, ct_heavy_idx, ct_light_idx, c_h_idx, c_l_idx):
                new_x[ct_heavy_idx,c_h_idx]=0
                new_x[ct_heavy_idx,c_l_idx]=1
                new_x[ct_light_idx,c_h_idx]=1
                new_x[ct_light_idx,c_l_idx]=0
                return new_x
    return new_x

def move_partial_cargo_to_least_loaded_container(eval_result:EvaluationResult)->Tuple[bool, np.ndarray]:
    cargo_to_move_idx = find_cargo_to_move_with_threshold(eval_result)
    if len(cargo_to_move_idx)==0:
        return False, None
    container_idx = find_least_loaded_container(eval_result, cargo_to_move_idx)
    if container_idx == -1:
        return False, None
    new_x = eval_result.x.copy()
    new_x[:, cargo_to_move_idx] = 0
    new_x[container_idx,cargo_to_move_idx] = 1
    return True, new_x
        
        

def move_heaviest_cargo_to_second_least_loaded_container(eval_result:EvaluationResult)->int:
    heaviest_infeasible_cargo_idx = find_heaviest_infeasible_cargos(eval_result)
    container_idx = find_second_least_loaded_container(eval_result, heaviest_infeasible_cargo_idx)
    if container_idx == -1:
        return False, None
    new_x = eval_result.x.copy()
    new_x[:, heaviest_infeasible_cargo_idx] = 0
    new_x[container_idx,heaviest_infeasible_cargo_idx] = 1
    return True, new_x

def move_heaviest_cargo_to_least_loaded_container(eval_result:EvaluationResult)->int:
    heaviest_infeasible_cargo_idx = find_heaviest_infeasible_cargos(eval_result)
    container_idx = find_least_loaded_container(eval_result, heaviest_infeasible_cargo_idx)
    if container_idx == -1:
        return False, None
    new_x = eval_result.x.copy()
    new_x[:, heaviest_infeasible_cargo_idx] = 0
    new_x[container_idx,heaviest_infeasible_cargo_idx] = 1
    return True, new_x

def move_heaviest_cargo_to_random_container(eval_result:EvaluationResult)->Tuple[bool, np.ndarray]:
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