from typing import List, Tuple

import numpy as np


from solver.utils import visualize_box
from heuristic.utils import insert_cargo_to_container
from heuristic.insertion.extreme_point.utils import project_extreme_point, init_insertion_points, find_ip_and_cargo_idx, argsort_cargo
from solver.solution import Solution
from solver.utils import get_possible_rotation_mats


"""
    Input:
        Solution
        List of cargo idx want to be inserted

    Output:
        the updated solution,
        the list of idxs that cannot/failed to be inserted
"""
def insert_many_cargo_to_one(solution: Solution,
                       cargo_idxs: List[int],
                       container_idx: int,
                       cargo_sort:str="volume-height",
                       match_method:str="first-fit") -> Tuple[Solution,List[int]]:
    # re-generate insertion points from existing cargos in the container
    container_dim = solution.container_dims[container_idx, :]
    container_cargo_idxs = np.argwhere(solution.cargo_container_maps==container_idx).flatten()
    insertion_points = np.zeros([1,3])
    cc_dims = solution.cargo_dims[container_cargo_idxs,:]
    cc_rotation_mats = solution.rotation_mats[container_cargo_idxs,:]
    cc_positions = solution.positions[container_cargo_idxs,:]
    cc_real_dims = (cc_dims[:,np.newaxis,:]*cc_rotation_mats).sum(axis=-1)
    if len(container_cargo_idxs) > 0:
        insertion_points = init_insertion_points(container_dim, cc_real_dims, cc_positions)

    # here we have to repeat all the cargos to be inserted
    # with all their possible rotations.
    # this includes repeating every attributes
    num_cargos_to_insert = len(cargo_idxs)
    possible_rotation_mats = get_possible_rotation_mats()
    c_rotation_mats = np.tile(possible_rotation_mats, (num_cargos_to_insert,1,1))
    c_real_idxs = np.repeat(cargo_idxs,6)
    c_weights = np.repeat(solution.cargo_weights[cargo_idxs], 6)
    c_volumes = np.repeat(solution.cargo_volumes[cargo_idxs], 6)
    c_dims = np.repeat(solution.cargo_dims[cargo_idxs,:],6,axis=0)
    c_real_dims = (c_dims[:,np.newaxis,:]*c_rotation_mats).sum(axis=-1)
    sorted_idx = argsort_cargo(c_real_dims, c_weights, c_volumes)
    c_rotation_mats = c_rotation_mats[sorted_idx]
    c_real_idxs = c_real_idxs[sorted_idx]
    c_weights = c_weights[sorted_idx]
    c_volumes = c_volumes[sorted_idx]
    c_real_dims = c_real_dims[sorted_idx]

    while True:
        ip_sorted_idx = np.lexsort((insertion_points[:, 0],  insertion_points[:, 1], insertion_points[:, 2]), axis=0)
        insertion_points = insertion_points[ip_sorted_idx]
        filled_weight = solution.container_filled_weights[container_idx]
        max_weight = solution.container_max_weights[container_idx]
        container_cargo_idxs = np.argwhere(solution.cargo_container_maps==container_idx).flatten()
        cc_dims = solution.cargo_dims[container_cargo_idxs,:]
        cc_rotation_mats = solution.rotation_mats[container_cargo_idxs,:]
        cc_positions = solution.positions[container_cargo_idxs,:]
        cc_real_dims = (cc_dims[:,np.newaxis,:]*cc_rotation_mats).sum(axis=-1)
        
        ip_idx, chosen_idx = find_ip_and_cargo_idx(container_dim,cc_real_dims, cc_positions, filled_weight, max_weight, insertion_points, c_real_dims, c_weights, c_volumes, match_method="first-fit")
        if ip_idx is None:
            break
        # insert cargo
        solution = insert_cargo_to_container(solution,c_real_idxs[chosen_idx], container_idx, c_rotation_mats[chosen_idx], insertion_points[ip_idx])
        # update insertion points
        ip = c_real_dims[[chosen_idx],:]*np.eye(3,3)+insertion_points[[ip_idx],:]
        ip = np.repeat(ip,2,axis=0)
        ip[0,:] = project_extreme_point(ip[0,:], container_dim, cc_real_dims, cc_positions, axis=1)
        ip[1,:] = project_extreme_point(ip[1,:], container_dim, cc_real_dims, cc_positions, axis=2)
        ip[2,:] = project_extreme_point(ip[2,:], container_dim, cc_real_dims, cc_positions, axis=0)
        ip[3,:] = project_extreme_point(ip[3,:], container_dim, cc_real_dims, cc_positions, axis=2)
        ip[4,:] = project_extreme_point(ip[4,:], container_dim, cc_real_dims, cc_positions, axis=0)
        ip[5,:] = project_extreme_point(ip[5,:], container_dim, cc_real_dims, cc_positions, axis=1)
        insertion_points = np.delete(insertion_points, ip_idx, 0)
        insertion_points = np.concatenate([insertion_points, ip])
        insertion_points = np.unique(insertion_points, axis=0)
        
        # update the attributes/ remove the chosen ones
        chosen_real_idx = c_real_idxs[chosen_idx]
        is_not_chosen = c_real_idxs != chosen_real_idx
        c_rotation_mats = c_rotation_mats[is_not_chosen]
        c_real_idxs = c_real_idxs[is_not_chosen]
        c_weights = c_weights[is_not_chosen]
        c_volumes = c_volumes[is_not_chosen]
        c_real_dims = c_real_dims[is_not_chosen]
    return solution