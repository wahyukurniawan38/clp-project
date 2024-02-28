from typing import Union, Tuple

import numba as nb
import numpy as np

from solver.problem import Problem
from solver.solution import SolutionBase

def add_container(solution:SolutionBase, container_type_idx):
    container_type_list = solution.problem.container_type_list
    container_type = container_type_list[container_type_idx]
    solution.container_dims = np.append(solution.container_dims, container_type.dim[np.newaxis,:], axis=0)
    solution.container_max_weights = np.append(solution.container_max_weights, [container_type.max_weight])
    # print(solution.container_filled_weights)
    solution.container_max_volumes = np.append(solution.container_max_volumes, [container_type.max_volume]) 
    # print(solution.container_filled_weights)
    solution.container_filled_volumes = np.append(solution.container_filled_volumes, [0])
    solution.container_filled_weights = np.append(solution.container_filled_weights, [0])
    solution.nums_container_used[container_type_idx] += 1
    solution.container_costs = np.append(solution.container_costs, [solution.problem.container_type_list[container_type_idx].cost])
    solution.container_cogs = np.append(solution.container_cogs, [[0,0]],axis=0)
    solution.container_cog_tolerances = np.append(solution.container_cog_tolerances, [solution.problem.container_type_list[container_type_idx].cog_tolerance])
    solution.container_types = np.append(solution.container_types, [container_type_idx])
    return solution


"""
    solution: solver.solution
    cargo_idx: scalar or np.ndarray(n,)
    container_idx: int
    rotation_mat: if cargo_idx int-> (3,3), else (n,3,3)
    position: if cargo_idx int -> (3,) else (n,3)
"""
def insert_cargo_to_container(solution:SolutionBase, 
                                cargo_idx:Union[int, np.ndarray], 
                                container_idx: int,
                                rotation_mat: np.ndarray, 
                                position: np.ndarray):
    solution.cargo_container_maps[cargo_idx] = container_idx
    solution.rotation_mats[cargo_idx,:] = rotation_mat
    solution.positions[cargo_idx,:] = position
    cargo_weight = solution.cargo_weights[cargo_idx]
    cargo_dim = solution.cargo_dims[cargo_idx]
    cargo_real_dim = (cargo_dim[np.newaxis,:]*rotation_mat).sum(axis=-1)
    old_weight = solution.container_filled_weights[container_idx]
    
    # recompute new center of gravity
    if np.any(solution.container_cogs[container_idx] <=0):
        solution.container_cogs[container_idx] = position[:2]+cargo_real_dim[:2]/2
    else:
        old_cog = solution.container_cogs[container_idx]
        new_cog = (old_cog*old_weight + cargo_weight*(position[:2]+cargo_real_dim[:2]/2))/(old_weight+cargo_weight)
        solution.container_cogs[container_idx] = new_cog

    solution.container_filled_volumes[container_idx] += solution.cargo_volumes[cargo_idx]
    solution.container_filled_weights[container_idx] += cargo_weight
    return solution


"""
    solution: Solution
    cargo_idx: (n,)
    container_idx: int
    # we should check if these cargos 
    really are in container idx,
    in this function, it assumes they are in the
    container
"""
def remove_cargos_from_container(solution: SolutionBase,
                                 cargo_idx: np.ndarray,
                                 container_idx: int)-> SolutionBase:
    if len(cargo_idx)==0:
        return solution
    total_removed_weight = np.sum(solution.cargo_weights[cargo_idx])
    total_removed_volume = np.sum(solution.cargo_volumes[cargo_idx])
    # re-arranging the center of gravity after removal
    old_cog = solution.container_cogs[container_idx]
    old_weight = solution.container_filled_weights[container_idx]

    c_real_dims = (solution.cargo_dims[cargo_idx,np.newaxis,:]*solution.rotation_mats[cargo_idx,:,:]).sum(axis=-1)
    c_centers = solution.positions[cargo_idx,:2] + c_real_dims[:,:2]/2
    c_weights = solution.cargo_weights[cargo_idx]

    solution.container_cogs[container_idx] = (old_cog*old_weight - np.sum(c_centers*c_weights[:, np.newaxis], axis=0))/max(old_weight-total_removed_weight,1)
    np.clip(solution.container_cogs, a_min=0, a_max=None)
    solution.cargo_container_maps[cargo_idx] = -1
    solution.positions[cargo_idx,:] = -1
    solution.container_filled_volumes[container_idx] -= total_removed_volume
    solution.container_filled_weights[container_idx] -= total_removed_weight
    return solution

"""
    solution: Solution
    container_idx: int
"""
def empty_container(solution: SolutionBase,
                    container_idx: int)-> SolutionBase:
    is_cargo_in_container = solution.cargo_container_maps == container_idx
    cargo_in_container_idx = np.nonzero(is_cargo_in_container)[0]
    if len(cargo_in_container_idx) > 0:
        solution = remove_cargos_from_container(solution, cargo_in_container_idx, container_idx)
    return solution

def remove_infeasible_cargo(solution: SolutionBase,
                            container_idx: int) -> Tuple[SolutionBase, np.ndarray]:
    infeasible_cargo_idx = []
    cc_unsopperted_idx = get_unsupported_cargo_idx_from_container(solution, container_idx)
    while len(cc_unsopperted_idx) > 0:
        infeasible_cargo_idx += [cc_unsopperted_idx]
        solution = remove_cargos_from_container(solution, cc_unsopperted_idx, container_idx)
        cc_unsopperted_idx = get_unsupported_cargo_idx_from_container(solution, container_idx)
    if len(infeasible_cargo_idx)>0:
        infeasible_cargo_idx = np.concatenate(infeasible_cargo_idx)
    return solution, infeasible_cargo_idx

def get_unsupported_cargo_idx_from_container(solution: SolutionBase,
                                            container_idx: int) -> np.ndarray:
    is_cargo_in_container = solution.cargo_container_maps == container_idx
    cc_idx = np.nonzero(is_cargo_in_container)[0]
    if len(cc_idx) == 0:
        return cc_idx
    cc_positions = solution.positions[cc_idx, :]
    cc_dims = solution.cargo_dims[cc_idx, :]
    cc_rotation_mats = solution.rotation_mats[cc_idx,:,:]
    cc_real_dims =  (cc_dims[:,np.newaxis,:]*cc_rotation_mats).sum(axis=-1)
    container_dim = solution.container_dims[container_idx, :]

    cc_bot_pos, cc_bot_dim = get_bottom_surface(cc_positions, cc_real_dims)
    container_bottom_pos, container_bottom_dim = get_bottom_surface(np.asanyarray([[0,0,0]]), container_dim[np.newaxis,:])
    cc_top_pos, cc_top_dim = get_top_surface(cc_positions, cc_real_dims)
    cc_top_pos = np.concatenate([cc_top_pos, container_bottom_pos], axis=0)
    cc_top_dim = np.concatenate([cc_top_dim, container_bottom_dim])
    is_on_top = cc_bot_pos[:,np.newaxis,2] == cc_top_pos[np.newaxis,:,2]
    base_support_area = compute_collision(cc_bot_pos[:,:2], cc_bot_dim[:,:2], cc_top_pos[:,:2], cc_top_dim[:,:2])
    base_support_area *= is_on_top
    base_support_area = np.sum(base_support_area, axis=-1)
    base_area = cc_real_dims[:,0]*cc_real_dims[:,1]
    supported_base_area_ratio = base_support_area/base_area
    is_base_supported = supported_base_area_ratio>0.5
    
    cc_unsopperted_idx = cc_idx[np.logical_not(is_base_supported)]
    return cc_unsopperted_idx

"""
    pos1: (n1x3)
    dim1: (n1x3)
    pos2: (n2x3)
    dim2: (n2x3)

    output: boolean matrix is collision happens between each pair
    out: (n1xn2)
"""
@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:]), parallel=True)
def is_collide_3d(pos1: np.ndarray, 
                  dim1: np.ndarray,
                  pos2: np.ndarray,
                  dim2: np.ndarray) -> np.ndarray:
    cp1 = pos1+dim1
    cp2 = pos2+dim2
    n, _ = pos1.shape
    m, _ = pos2.shape
    # print(n,m)
    is_collide = np.ones((n,m))
    for i in nb.prange(n):
        for j in range(m):
            if np.any(pos1[i,:]>=cp2[j,:]) or np.any(cp1[i,:]<=pos2[j,:]):
                is_collide[i,j] = 0
    # is_not_collide1 = np.sum(pos1[:,np.newaxis,:] >= cp2[np.newaxis,:,:], axis=2) > 0
    # is_not_collide2 = np.sum(cp1[:,np.newaxis,:] <= pos2[np.newaxis,:,:], axis=2) > 0
    # is_not_collide = np.logical_or(is_not_collide1, is_not_collide2)
    # assert np.allclose(is_not_collide, is_not_collide_)
    # is_collide = np.logical_not(is_not_collide)
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

def get_feasibility_mask(container_dim:np.ndarray,
                          cc_dims:np.ndarray, 
                          cc_positions:np.ndarray, 
                          cc_filled_weight: float,
                          cc_max_weight: float,
                          insertion_points:np.ndarray, 
                          c_dims:np.ndarray, 
                          c_weights:np.ndarray, 
                          c_volumes:np.ndarray):
    n_cargo = len(c_dims)
    n_ip = len(insertion_points)
    # the weight capacity can fit the cargo
    is_weight_cap_enough = c_weights + cc_filled_weight <= cc_max_weight
    is_weight_cap_enough = np.repeat(is_weight_cap_enough, n_ip, axis=0)
    # try all possible insertion positions
    # for all cargo
    # and check if it collides with cargos already in the container
    # repeat insertion points n_cargo times
    # repeat cargo dims n_insert_points times
    c_dims_ = np.repeat(c_dims, n_ip, axis=0)
    insertion_points_ = np.tile(insertion_points, [n_cargo,1])
    is_collide_all = is_collide_3d(insertion_points_, c_dims_, cc_positions, cc_dims)
    is_collide_with_any = np.any(is_collide_all, axis=1)
    is_not_collide_with_any = np.logical_not(is_collide_with_any)

    # check if overflow the container
    is_not_overflow = (c_dims_ + insertion_points_) <= container_dim[np.newaxis,:]
    is_not_overflow = np.all(is_not_overflow, axis=-1)
    # combine all
    feasibility_mask = np.logical_and(is_not_collide_with_any, is_weight_cap_enough)
    feasibility_mask = np.logical_and(feasibility_mask, is_not_overflow)
    feasibility_mask = feasibility_mask.reshape([n_cargo, n_ip])
    return feasibility_mask