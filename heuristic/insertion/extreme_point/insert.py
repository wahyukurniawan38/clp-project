from typing import List, Tuple

import numpy as np

from heuristic.utils import insert_cargo_to_container
from heuristic.insertion.extreme_point.utils import init_insertion_points, find_ip_and_cargo_idx
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
                       container_idx: int) -> Tuple[Solution,List[int]]:
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
     
    is_insertable = True
    while is_insertable:
    
        ip_idx, chosen_cargo_idx = find_ip_and_cargo_idx(cc_real_dims, cc_positions, insertion_points, c_real_dims, c_weights, c_volumes, cargo_sort="volume-height", ip_sort="first-fit")
        insertion_points = update_insertion_points(insertion_points, cc_real_dims)
        solution = insert_cargo_to_container

