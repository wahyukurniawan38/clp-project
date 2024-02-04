from typing import List, Tuple

import numpy as np

from heuristic.utils import insert_cargo_to_container
from heuristic.insertion.extreme_point.utils import init_insertion_points, find_ip_and_cargo_idx, argsort_cargo
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
    insertion_points = np.empty([0,3], dtype=float)
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
    # for uniqueness we need this temp idx
    c_temp_idxs = np.arange(len(cargo_idxs))
    c_temp_idxs = np.repeat(c_temp_idxs, 6)
    sorted_idx = argsort_cargo(c_real_dims, c_weights)
    filled_weight = solution.container_filled_weights[container_idx]
    max_weight = solution.container_max_weights[container_idx]
    is_insertable = True
    while is_insertable:
        
        ip_idx, chosen_idx = find_ip_and_cargo_idx(container_dim,cc_real_dims, cc_positions, filled_weight, max_weight, insertion_points, c_real_dims, c_weights, c_volumes, match_method="first-fit")
        # insertion_points = update_insertion_points(insertion_points, cc_real_dims)
        solution = insert_cargo_to_container

