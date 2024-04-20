import numpy as np

from heuristic.alns_wahyu.evaluator.evaluator import EvaluationResult, Evaluator
from solver.problem import Problem
from solver.solution import SolutionBase


def remove_unpacked_cargo_from_solution(solution: SolutionBase)->SolutionBase:
    unpacked_cargo_idx = np.where(solution.cargo_container_maps<0)[0]
    packed_cargo_mask = solution.cargo_container_maps>=0
    unpacked_cargo_ids = solution.cargo_type_ids[unpacked_cargo_idx]
    container_type_list = solution.problem.container_type_list
    cargo_type_list = [ctype for ctype in solution.problem.cargo_type_list if ctype.id not in unpacked_cargo_ids]
    new_problem = Problem(cargo_type_list, container_type_list)

    solution.problem = new_problem
    solution.cargo_dims =  new_problem.cargo_dims
    solution.cargo_types = new_problem.cargo_types
    solution.cargo_weights = new_problem.cargo_weights
    solution.cargo_costs = new_problem.cargo_costs
    solution.cargo_volumes = new_problem.cargo_volumes
    solution.cargo_type_ids = np.asanyarray([ct.id for ct in new_problem.cargo_type_list]).astype(int)

    solution.positions = solution.positions[packed_cargo_mask,:]
    solution.cargo_container_maps = solution.cargo_container_maps[packed_cargo_mask]
    solution.rotation_mats = solution.rotation_mats[packed_cargo_mask,:,:]
    return solution

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


def find_heaviest_infeasible_cargos(eval_result:EvaluationResult)->EvaluationResult:
    cargo_volumes = eval_result.cargo_volumes
    cargo_weights = eval_result.cargo_weights
    x = eval_result.x
    infeasible_container_idx = [i for i, solution in enumerate(eval_result.solution_list) if (not solution.is_cog_feasible or not solution.is_all_cargo_packed)]
    infeasible_container_idx = np.array(infeasible_container_idx)
    weights_in_infeasible_container = x[infeasible_container_idx,:]*cargo_weights[np.newaxis,:]
    heaviest_items = np.argmax(weights_in_infeasible_container,axis=1)
    return heaviest_items


def get_container_capacity_feasibility_mask(eval_result:EvaluationResult, 
                                            cargo_to_insert_idx:np.ndarray=None)->np.ndarray:
    if cargo_to_insert_idx is None:
        return np.asanyarray([True for i in range(len(eval_result.x))], dtype=bool)
    cargo_volumes = eval_result.cargo_volumes
    cargo_weights = eval_result.cargo_weights
    hi_volume = np.sum(cargo_volumes[cargo_to_insert_idx])
    hi_weight = np.sum(cargo_weights[cargo_to_insert_idx])
    max_volume = eval_result.max_container_volume
    max_weight = eval_result.max_container_weight
    # remove the cargo_to_insert_idx first
    x_ = eval_result.x.copy()
    x_[:, cargo_to_insert_idx] = 0
    container_filled_volume = x_.dot(cargo_volumes)
    container_filled_weight = x_.dot(cargo_weights)
    remaining_container_volumes = max_volume-container_filled_volume
    remaining_container_weights = max_weight-container_filled_weight
    container_has_enough_weight_cap = hi_weight<=remaining_container_weights
    container_has_enough_vol_cap = hi_volume<=remaining_container_volumes
    is_cap_feasible = np.logical_and(container_has_enough_vol_cap, container_has_enough_weight_cap)
    return is_cap_feasible


def find_second_least_loaded_container(eval_result:EvaluationResult,
                                cargo_to_insert_idx:np.ndarray=None)->int:
    cargo_loads = eval_result.cargo_loads
    is_container_filled = np.any(eval_result.x, axis=1)
    container_load = eval_result.x.dot(cargo_loads)
    container_sorted_idx = np.argsort(container_load)
    is_cap_feasible = get_container_capacity_feasibility_mask(eval_result, cargo_to_insert_idx)
    if not np.any(is_cap_feasible):
        return -1
    is_cap_feasible = is_cap_feasible[container_sorted_idx]
    is_container_filled = is_container_filled[container_sorted_idx]
    if np.any(np.logical_and(is_container_filled, is_cap_feasible)):
        container_sorted_idx = container_sorted_idx[np.logical_and(is_container_filled, is_cap_feasible)]
    else:
        container_sorted_idx = container_sorted_idx[is_cap_feasible]
    if len(container_sorted_idx)<2:
        return -1
    return container_sorted_idx[1]

def find_least_loaded_container(eval_result:EvaluationResult,
                                cargo_to_insert_idx:np.ndarray=None)->int:
    cargo_loads = eval_result.cargo_loads
    is_container_filled = np.any(eval_result.x, axis=1)
    container_load = eval_result.x.dot(cargo_loads)
    container_sorted_idx = np.argsort(container_load)
    is_cap_feasible = get_container_capacity_feasibility_mask(eval_result, cargo_to_insert_idx)
    if not np.any(is_cap_feasible):
        return -1
    is_cap_feasible = is_cap_feasible[container_sorted_idx]
    is_container_filled = is_container_filled[container_sorted_idx]
    if np.any(np.logical_and(is_container_filled, is_cap_feasible)):
        container_sorted_idx = container_sorted_idx[np.logical_and(is_container_filled, is_cap_feasible)]
    else:
        container_sorted_idx = container_sorted_idx[is_cap_feasible]
    return container_sorted_idx[0]

def find_most_loaded_container(eval_result:EvaluationResult,
                                cargo_to_insert_idx:np.ndarray=None)->int:
    cargo_loads = eval_result.cargo_loads
    is_container_filled = np.any(eval_result.x, axis=1)
    container_load = eval_result.x.dot(cargo_loads)
    container_sorted_idx = np.argsort(container_load)
    is_cap_feasible = get_container_capacity_feasibility_mask(eval_result, cargo_to_insert_idx)
    if not np.any(is_cap_feasible):
        return -1
    is_cap_feasible = is_cap_feasible[container_sorted_idx]
    is_container_filled = is_container_filled[container_sorted_idx]
    if np.any(np.logical_and(is_container_filled, is_cap_feasible)):
        container_sorted_idx = container_sorted_idx[np.logical_and(is_container_filled, is_cap_feasible)]
    else:
        container_sorted_idx = container_sorted_idx[is_cap_feasible]
    return container_sorted_idx[-1]


def find_suitable_random_container(eval_result:EvaluationResult, 
                                   cargo_to_insert_idx:np.ndarray)->int:
    is_cap_feasible = get_container_capacity_feasibility_mask(eval_result, cargo_to_insert_idx)
    if not np.any(is_cap_feasible):
        return -1
    suitable_containers = np.where(is_cap_feasible)[0]
    selected_container = np.random.choice(suitable_containers)
    return selected_container


def find_cargo_to_move_with_threshold(eval_result:EvaluationResult)->np.ndarray:
    is_container_feasible = np.asanyarray([(solution.is_cog_feasible and solution.is_all_cargo_packed) for solution in eval_result.solution_list])
    infeasible_container_idx = np.where(np.logical_not(is_container_feasible))[0]
    cargo_loads = eval_result.cargo_loads
    cargo_to_move_idx = []
    x = eval_result.x
    for ct_idx in infeasible_container_idx:
        cc_idx = np.where(x[ct_idx,:])[0]
        cc_load = cargo_loads[cc_idx]
        container_load = np.sum(cc_load)
        load_sorted_idx = np.argsort(cc_load)
        cc_load = cc_load[load_sorted_idx]
        cc_sorted_idx = cc_idx[load_sorted_idx]
        cc_cum_load = np.cumsum(cc_load)
        remaining_load_threshold = np.random.uniform(0.4, 0.75)*container_load
        remaining_load_after_removal = container_load - cc_cum_load
        is_cargo_chosen = remaining_load_after_removal>=remaining_load_threshold
        chosen_cargo_idx = cc_sorted_idx[is_cargo_chosen]
        cargo_to_move_idx += [chosen_cargo_idx]
    if len(cargo_to_move_idx)>0:
        cargo_to_move_idx = np.concatenate(cargo_to_move_idx)
    return cargo_to_move_idx

def can_swap(eval_result: EvaluationResult, 
             ct_a: int, 
             ct_b: int, 
             c_a: int, 
             c_b: int)->bool:
    x_tmp = eval_result.x.copy()
    x_tmp[ct_a,c_a]=0
    x_tmp[ct_a,c_b]=1
    x_tmp[ct_b,c_a]=1
    x_tmp[ct_b,c_b]=0
    cargo_volumes, cargo_weights = eval_result.cargo_volumes, eval_result.cargo_weights
    max_container_vol, max_container_weight = eval_result.max_container_volume, eval_result.max_container_weight
    filled_vol_after_swap = x_tmp.dot(cargo_volumes)
    filled_weight_after_swap = x_tmp.dot(cargo_weights)
    ct_a_vol_after_swap = filled_vol_after_swap[ct_a]
    if ct_a_vol_after_swap > max_container_vol:
        return False
    ct_b_vol_after_swap = filled_vol_after_swap[ct_b]
    if ct_b_vol_after_swap > max_container_vol:
        return False
    ct_a_weight_after_swap = filled_weight_after_swap[ct_a]
    if ct_a_weight_after_swap > max_container_weight:
        return False
    ct_b_weight_after_swap = filled_weight_after_swap[ct_b]
    if ct_b_weight_after_swap > max_container_weight:
        return False
    return True