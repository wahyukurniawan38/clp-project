from typing import Tuple

import numpy as np

from heuristic.utils import get_feasibility_mask, insert_cargo_to_container
from heuristic.lns_safak.solution import Solution
from heuristic.lns_safak.utils import get_addition_points, argsort_addition_points, filter_infeasible_addition_points
from solver.utils import get_possible_rotation_mats

def add_item_to_container(solution: Solution,
                          cargo_idx: np.ndarray,
                          container_idx: int,
                          mode:str="wall-building")-> Tuple[Solution, np.ndarray]:
    is_cargo_in_container = solution.cargo_container_maps == container_idx
    cc_idx = np.nonzero(is_cargo_in_container)[0]
    cc_positions = solution.positions[cc_idx, :]
    cc_dims = solution.cargo_dims[cc_idx,:]
    cc_rotation_mats = solution.rotation_mats[cc_idx,:]
    cc_real_dims =  (cc_dims[:,np.newaxis,:]*cc_rotation_mats).sum(axis=-1)
    container_dim = solution.container_dims[container_idx]
    addition_points = get_addition_points(cc_positions, cc_real_dims, container_dim)
    sorted_ap_idx = argsort_addition_points(addition_points, mode)
    addition_points = addition_points[sorted_ap_idx]


    # order cargo by the preset cargo type priority
    c_type = solution.cargo_types[cargo_idx]
    c_priority = solution.cargo_type_priority[c_type]
    c_sorted_idx = np.argsort(c_priority)
    cargo_idx = cargo_idx[c_sorted_idx]
    c_dims = solution.cargo_dims[cargo_idx, :]
    c_type = solution.cargo_types[cargo_idx]
    c_rotation_sorted_idx = solution.cargo_type_rotation_sorted_idx[c_type]
    num_cargos_to_insert = len(cargo_idx)
    possible_rotation_mats = get_possible_rotation_mats()
    is_inserted = np.zeros([num_cargos_to_insert,], dtype=bool)
    for i in range(num_cargos_to_insert):
        c_rotation_mats = possible_rotation_mats[c_rotation_sorted_idx[i],:, :]
        ci_dims = np.repeat(c_dims[[i],:], 6, axis=0)
        ci_real_dims = (ci_dims[:,np.newaxis,:]*c_rotation_mats).sum(axis=-1)
        
        ci_weights = np.repeat(solution.cargo_weights[i], 6)
        ci_volumes = np.repeat(solution.cargo_volumes[i], 6)
        feasibility_mask = get_feasibility_mask(container_dim,
                                                cc_real_dims,
                                                cc_positions,
                                                solution.container_filled_weights[container_idx],
                                                solution.container_max_weights[container_idx],
                                                addition_points,
                                                ci_real_dims,
                                                ci_weights,
                                                ci_volumes)
        feasible_rotation_idx, feasible_pos_idx = np.nonzero(feasibility_mask)
        if len(feasible_rotation_idx)==0:
            continue
        chosen_cargo_idx = cargo_idx[i]
        chosen_rotation_idx = feasible_rotation_idx[0]
        chosen_pos_idx = feasible_pos_idx[0]
        chosen_r = c_rotation_mats[chosen_rotation_idx]
        chosen_p = addition_points[chosen_pos_idx]

        solution = insert_cargo_to_container(solution, chosen_cargo_idx, container_idx, chosen_r, chosen_p)
        is_inserted = True
        
        # update the cargos in the container
        cc_positions = np.concatenate([cc_positions, chosen_p[np.newaxis, :]], axis=0)
        cc_rotation_mats = np.concatenate([cc_rotation_mats, chosen_r[np.newaxis, :, :]], axis=0)
        cc_real_dims =  np.concatenate([cc_real_dims, ci_real_dims[[chosen_rotation_idx],:]], axis=0)
        # add addition points
        new_addition_points = ci_real_dims[[chosen_rotation_idx],:]*np.eye(3,3) + chosen_p[np.newaxis, :]
        addition_points = np.concatenate([addition_points, new_addition_points], axis=0)
        addition_points = filter_infeasible_addition_points(addition_points, cc_positions, cc_real_dims, container_dim)
    
    is_not_inserted = np.logical_not(is_inserted)
    not_inserted_cargo_idx = cargo_idx[is_not_inserted]
    return solution, not_inserted_cargo_idx