from copy import deepcopy

import numpy as np
import pandas as pd

from heuristic.alns_wahyu.evaluator.evaluator import Evaluator
from heuristic.alns_wahyu.evaluator.evaluation_result import EvaluationResult, create_copy
from heuristic.alns_wahyu.utils import remove_unpacked_cargo_from_solution
from solver.solution import SolutionBase

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

def remove_unpacked_cargo_from_result(eval_result:EvaluationResult)->EvaluationResult:
    failed_cargo_ids = []
    for s_idx, solution in enumerate(eval_result.solution_list):
        if solution.is_all_cargo_packed:
            continue
        failed_cargo_ids += [solution.cargo_type_ids[solution.cargo_container_maps==-1]]
        eval_result.solution_list[s_idx] = remove_unpacked_cargo_from_solution(solution)
    if len(failed_cargo_ids)==0:
        return eval_result
    failed_cargo_ids = np.concatenate(failed_cargo_ids)
    new_x = eval_result.x.copy()
    new_x[:,failed_cargo_ids] = 0
    eval_result.x = new_x
    return eval_result


# here is the recursion in step number three
def insert_unpacked_cargo_to_containers(eval_result: EvaluationResult,
                                        evaluator: Evaluator,
                                       allowed_container_idx: np.ndarray)->EvaluationResult:
    if len(allowed_container_idx)==0:
        return eval_result
    unpacked_cargo_idx = np.where(np.logical_not(np.any(eval_result.x, axis=0)))[0]
    if len(unpacked_cargo_idx)==0:
        return eval_result
    chosen_container_idx = allowed_container_idx[0]
    # try packing again
    new_x = eval_result.x.copy()
    new_x[chosen_container_idx,unpacked_cargo_idx] = 1
    df_cargos, df_containers = eval_result.df_cargos, eval_result.df_containters
    new_solution = evaluator.solve(new_x[chosen_container_idx,:], df_cargos, df_containers)
    eval_result.solution_list[chosen_container_idx] = new_solution
    eval_result.x = new_x
    eval_result = remove_unpacked_cargo_from_result(eval_result)
    return insert_unpacked_cargo_to_containers(eval_result, evaluator, allowed_container_idx[1:])    


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
    solution.positions[:,2] += possible_shift[np.newaxis,:]
    solution.container_cogs[0,:] += possible_shift
    return solution