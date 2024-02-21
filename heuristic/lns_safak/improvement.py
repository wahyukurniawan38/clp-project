import random
from typing import Tuple

import numpy as np

from heuristic.lns_safak.constructive import constructive_heuristic
from heuristic.lns_safak.solution import Solution, create_copy
from heuristic.lns_safak.insert import add_item_to_container
from solver.objective import is_better, compute_objective
from heuristic.utils import empty_container, remove_cargos_from_container, remove_infeasible_cargo

def improvement_heuristic(current_solution:Solution,
                          best_solution:Solution,
                          g: float = 0.2,
                          ld: float = 0.2,
                          max_perturbation_iter = 10,
                          max_repair_iter = 10,
                          mode="layer-building")->Tuple[Solution,Solution]:
    for t in range(max_perturbation_iter):
        if random.random() < g:
            current_solution = create_copy(best_solution)
    
        current_solution = perturb_priority(current_solution)
        for ct_idx in range(len(current_solution.container_dims)):
            thr = (1-(current_solution.container_filled_volumes[ct_idx]/current_solution.container_max_volumes[ct_idx]))/2
            if random.random() <= thr:
                current_solution = empty_container(current_solution, ct_idx)
            else:
                packed_cargo_idx = np.nonzero(current_solution.cargo_container_maps==ct_idx)[0]
                num_to_remove = random.randint(0,len(packed_cargo_idx)//2)
                cargo_to_remove = np.random.choice(packed_cargo_idx, [num_to_remove], replace=False)
                current_solution = remove_cargos_from_container(current_solution, cargo_to_remove, ct_idx)
                current_solution, infeasible_cargo_idx = remove_infeasible_cargo(current_solution, ct_idx)
                if len(infeasible_cargo_idx)>0:
                    current_solution, failed_to_insert_cargo_idx = add_item_to_container(current_solution, infeasible_cargo_idx, ct_idx, mode)
        current_solution = constructive_heuristic(current_solution, mode)
        if not best_solution.is_feasible and current_solution.is_feasible:
            best_solution = create_copy(current_solution)
        elif is_better(current_solution, best_solution):
            best_solution = create_copy(current_solution)
        print("---------t:",t)
        print("Current:", compute_objective(current_solution))
        print("Best:", compute_objective(best_solution))
    return current_solution, best_solution        




def perturb_priority(solution:Solution, a=0.1, b=0.2)->Solution:
    r1 = np.random.random(solution.cargo_type_priority.shape)
    num_cargo_type = len(solution.cargo_type_priority)
    c_type_swap_idx = (np.random.randint(1,num_cargo_type,size=[num_cargo_type,]) + np.arange(num_cargo_type))%num_cargo_type
    is_swap_cargo_type_priority = r1<a

    for i in range(num_cargo_type):
        if is_swap_cargo_type_priority[i]:
            solution.cargo_type_priority[i] = solution.cargo_type_priority[c_type_swap_idx[i]]


    r2 = np.random.random(solution.cargo_type_rotation_sorted_idx.shape)
    is_swap_rotation_sorted_idx = r2<b
    idx = np.arange(6)[np.newaxis]
    c_type_rotation_swap_idx = (np.random.randint(1,6,size=[num_cargo_type,6]) + idx)%6
    for i in range(num_cargo_type):
        for j in range(6):
            if is_swap_rotation_sorted_idx[i,j]:
                j2 = c_type_rotation_swap_idx[i,j]
                solution.cargo_type_rotation_sorted_idx[i,j] = solution.cargo_type_rotation_sorted_idx[i,j2]
    return solution