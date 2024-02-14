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


"""
    pos1: (n1x3)
    dim1: (n1x3)
    pos2: (n2x3)
    dim2: (n2x3)

    output: boolean matrix is collision happens between each pair
    out: (n1xn2)
"""
def is_collide_3d(pos1: np.ndarray, 
                  dim1: np.ndarray,
                  pos2: np.ndarray,
                  dim2: np.ndarray) -> np.ndarray:
    cp1 = pos1+dim1
    cp2 = pos2+dim2
    is_not_collide1 = np.any(pos1[:,np.newaxis,:] >= cp2[np.newaxis,:,:], axis=2)
    is_not_collide2 = np.any(cp1[:,np.newaxis,:] <= pos2[np.newaxis,:,:], axis=2)
    is_not_collide = np.logical_or(is_not_collide1, is_not_collide2)
    is_collide = np.logical_not(is_not_collide)
    return is_collide

"""
    pos1: (n1xd)
    dim1: (n1xd)
    pos2: (n2xd)
    dim2: (n2xd)
    d = {1,2,3}
    output: matrix of collision area/volume of each pair
    out: (n1xn2)
"""
def compute_collision(pos1: np.ndarray, 
                        dim1: np.ndarray,
                        pos2: np.ndarray,
                        dim2: np.ndarray) -> np.ndarray:
    cp1 = pos1+dim1
    cp2 = pos2+dim2
    low_collision_pos = np.maximum(pos1[:, np.newaxis,:], pos2[np.newaxis,:,:])
    high_collision_pos = np.minimum(cp1[:, np.newaxis,:], cp2[np.newaxis,:,:])
    collision_width = high_collision_pos-low_collision_pos
    collision_area = np.prod(collision_width, axis=-1)
    collision_area = np.clip(collision_area, a_min=0, a_max=None)
    return collision_area

"""
    it's actually just simple indexing
    pos: nx3
    dim: nx3

    out:
        top_pos: nx3
        top_dim: nx3
"""
def get_top_surface(pos, dim):
    top_pos = pos + dim * np.asanyarray([[0,0,1]])
    top_dim = dim - dim * np.asanyarray([[0,0,1]])
    return top_pos, top_dim

"""
    it's actually just simple indexing
    pos: nx3
    dim: nx3

    out:
        bot_pos: nx3
        bot_dim: nx3
"""
def get_bottom_surface(pos, dim):
    bot_pos = pos
    bot_dim = dim - dim * np.asanyarray([[0,0,1]])
    return bot_pos, bot_dim

"""
    it's actually just simple indexing
    pos: nx3
    dim: nx3

    out:
        right_pos: nx3
        right_dim: nx3
"""
def get_right_surface(pos, dim):
    right_pos = pos + dim * np.asanyarray([[0,1,0]])
    right_dim = dim - dim * np.asanyarray([[0,1,0]])
    return right_pos, right_dim

"""
    it's actually just simple indexing
    pos: nx3
    dim: nx3

    out:
        front_pos: nx3
        front_dim: nx3
"""
def get_front_surface(pos, dim):
    front_pos = pos + dim * np.asanyarray([[1,0,0]])
    front_dim = dim - dim * np.asanyarray([[1,0,0]])
    return front_pos, front_dim
