import numpy as np

from solver.problem import Problem
from solver.solution import Solution

def add_container(solution:Solution, container_type_idx):
    container_type_list = solution.problem.container_type_list
    container_type = container_type_list[container_type_idx]
    solution.container_dims = np.append(solution.container_dims, container_type.dim[np.newaxis,:], axis=0)
    solution.container_max_volumes = np.append(solution.container_max_weights, [container_type.max_weight])
    solution.container_max_weights = np.append(solution.container_max_volumes, [container_type.max_volume]) 
    solution.container_filled_volumes = np.append(solution.container_filled_volumes, [0])
    solution.container_filled_weights = np.append(solution.container_filled_weights, [0])
    solution.nums_container_used[container_type_idx] += 1
    return solution

def insert_cargo_to_container(solution:Solution, 
                                cargo_idx, 
                                container_idx,
                                rotation_mat, 
                                position):
    solution.cargo_container_maps[cargo_idx] = container_idx
    solution.rotation_mats[cargo_idx,:] = rotation_mat
    solution.positions[cargo_idx,:] = position
    solution.container_filled_volumes[container_idx] += solution.cargo_volumes[cargo_idx]
    solution.container_filled_weights[container_idx] += solution.cargo_weights[cargo_idx]
    return solution


def check_collision_3d(p_min, p_max, q_min, q_max):
    if p_min[0] > q_max[0]:
        return False
    if p_max[0] < q_min[0]:
        return False
    if p_min[1] > q_max[1]:
        return False
    if p_max[1] < q_min[1]:
        return False
    if p_min[2] > q_max[2]:
        return False
    if p_max[2] < q_min[2]:
        return False
    return True

def check_collision_3d_vectorized(p_min, p_max, q_min, q_max):
    not_col1 = np.max(p_min > q_max, axis=1)
    not_col2 = np.max(p_max < q_min, axis=1)
    not_col = np.logical_or(not_col1, not_col2)
    is_collide = np.logical_not(not_col)
    return is_collide

def check_collision_3d_vectorized_multi_source(p_min, p_max, q_min, q_max):
    # print(p_min.shape. q_min.shape)
    # exit()
    not_col1 = np.any(p_min[:, np.newaxis,:] > q_max[np.newaxis,:,:], axis=2)
    not_col2 = np.any(p_max[:, np.newaxis,:] < q_min[np.newaxis,:,:], axis=2)
    not_col = np.logical_or(not_col1, not_col2)
    is_collide = np.logical_not(not_col)
    return is_collide

"""
    suppose there are 2d surfaces in 3d space
    this function computes their collision
    ofc, only ones that collide in xy axis AND has the same height (same z values) collide
"""
def compute_xy_collision_multi_source(p_s0:np.ndarray, p_s1:np.ndarray, z_s0:np.ndarray, z_s1:np.ndarray):
    p_s0,p_s1 = p_s0[:,np.newaxis,:], p_s1[:,np.newaxis,:] 
    z_s0,z_s1 = z_s0[np.newaxis,:,:], z_s1[np.newaxis,:,:] 
    a = np.maximum(p_s0,z_s0)
    b = np.minimum(p_s1,z_s1)
    diffs = b-a
    is_same_height = diffs[:,:,2]==0
    diffs = diffs[:,:,:2]
    diffs[diffs<0] = 0
    collision_area = np.prod(diffs, axis=-1)
    total_collision_area = np.sum(collision_area*is_same_height, axis=-1)
    return total_collision_area