import numpy as np

from solver.utils import get_possible_rotation_mats
from heuristic.lns_safak.solution import Solution




"""
    This is one of the two main operations
    in Safak LNS
    This operator try to pack every unpacked cargo
    into the container
    Ordering cargo is not in this method
    Ordering their rotation is in this method tho
"""
def constructive_heuristic(solution:Solution):
    is_cargo_unpacked = solution.cargo_container_maps == -1
    unpacked_cargo_idx = np.nonzero(is_cargo_unpacked)[0]
    unpacked_cargo_priority = solution.cargo_priority[is_cargo_unpacked]
    priority_sorted_idx = np.argsort(unpacked_cargo_priority)
    unpacked_cargo_idx = unpacked_cargo_idx[priority_sorted_idx]
    # sort container first
    container_filled_volumes = solution.container_filled_volumes
    container_costs = solution.container_costs
    sorted_container_idx = np.lexsort((-container_costs, -container_filled_volumes))
    for container_idx in sorted_container_idx:
        container_dim = solution.container_dims[container_idx]
        num_unpacked_cargo = len(unpacked_cargo_idx)
        c_dims = solution.cargo_dims[unpacked_cargo_idx]
        
        # sort their rotation first
        c_dims_ = np.repeat(c_dims, 6, axis=0)
        possible_rotation_mats = get_possible_rotation_mats()
        c_rotation_mats = np.tile(possible_rotation_mats, [num_unpacked_cargo,1,1])
        c_real_dims_ = (c_dims_[:,np.newaxis,:]*c_rotation_mats).sum(axis=-1)
        c_real_wall_dims_ = c_real_dims_[:, [0,1]]
        container_wall_dim = container_dim[[0,1]]
        num_wall_span = np.floor(container_wall_dim[np.newaxis,:]/ c_real_wall_dims_)
        wall_span_area = np.prod(np.floor(wall_span_area), axis=-1)
        print(c_real_wall_dims_.shape)
        print(container_wall_dim.shape)
        print(wall_span_area)
        exit()
        # update unpacked cargo idx




    exit()