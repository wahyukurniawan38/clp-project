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