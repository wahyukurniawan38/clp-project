from solver.problem import Problem

import numpy as np

from heuristic.utils import is_collide_3d, get_bottom_surface, get_top_surface, compute_collision
from solver.utils import get_possible_rotation_mats

def init_cargo_type_priority(problem:Problem, cargo_type_priority=None):
    if cargo_type_priority is not None:
        return cargo_type_priority.copy()
    cargo_type_priority = np.arange(len(problem.cargo_type_list), dtype=int)
    return cargo_type_priority


def init_cargo_type_rotation_sorted_idx(problem:Problem, cargo_type_rotation_sorted_idx=None):
    if cargo_type_rotation_sorted_idx is not None:
        return cargo_type_rotation_sorted_idx.copy()
    cargo_type_rotation_sorted_idx = np.arange(6)[np.newaxis,:]
    cargo_type_rotation_sorted_idx = np.repeat(cargo_type_rotation_sorted_idx, len(problem.cargo_type_list), axis=0)
    return cargo_type_rotation_sorted_idx


"""
    I assume sorting the rotation of each
    cargo type for each container type
    will be done very very often,
    that is why, let's pre-compute
    them, and fetch them when we need them.


    out:
        default_rotation: (n_cargo_type, n_container_type, 6)
"""
def init_default_cargo_type_rotation_sorted_idx(problem: Problem, default_cargo_type_rotation_sorted_idx=None):
    if default_cargo_type_rotation_sorted_idx is not None:
        return default_cargo_type_rotation_sorted_idx
    cargo_type_dims = [problem.cargo_type_list[i].dim[np.newaxis,:] for i in range(len(problem.cargo_type_list))]
    cargo_type_dims = np.concatenate(cargo_type_dims, axis=0)
    container_type_dims = [problem.container_type_list[i].dim[np.newaxis,:] for i in range(len(problem.container_type_list))]
    container_type_dims = np.concatenate(container_type_dims, axis=0)
    possible_rotation_mats = get_possible_rotation_mats()
    c_type_real_dims = (cargo_type_dims[:,np.newaxis,np.newaxis,:]*possible_rotation_mats[np.newaxis,:,:,:]).sum(axis=-1)
    c_type_real_wall_dims = c_type_real_dims[:,:,[1,2]]
    ct_type_wall_dims = container_type_dims[:,[1,2]]
    num_wall_span  = np.floor(ct_type_wall_dims[np.newaxis,np.newaxis,:,:]/c_type_real_wall_dims[:,:,np.newaxis,:])
    wall_span_area = np.prod(num_wall_span*c_type_real_wall_dims[:,:,np.newaxis,:],axis=-1)
    wall_span_area = np.transpose(wall_span_area, axes=(0,2,1))
    default_c_rotation_sorted_idx = np.argsort(wall_span_area,axis=-1)
    return default_c_rotation_sorted_idx

def filter_infeasible_addition_points(addition_points, cc_positions, cc_dims, container_dim):
    dummy_dim = np.asanyarray([[0.00001,0.00001,0.00001]]*len(addition_points))
    # if the addition point is inside other cargo
    # cargo already placed on it
    is_collide_matrix = is_collide_3d(addition_points, dummy_dim, cc_positions, cc_dims)
    is_collide_any = np.any(is_collide_matrix, axis=-1)
    is_not_collide = np.logical_not(is_collide_any)

    # if it is on the edges of the container
    is_not_overflow = (dummy_dim + addition_points) <= container_dim[np.newaxis,:]
    is_not_overflow = np.all(is_not_overflow, axis=-1)
    

    # if it is floating (not on the container base of a cargo's surface)
    c_bottom_pos_, c_bottom_dims_ = get_bottom_surface(addition_points, dummy_dim)
    container_bottom_pos, container_bottom_dim = get_bottom_surface(np.asanyarray([[0,0,0]]), container_dim[np.newaxis,:])
    cc_top_pos, cc_top_dim = get_top_surface(cc_positions, cc_dims)
    cc_top_pos = np.concatenate([cc_top_pos, container_bottom_pos], axis=0)
    cc_top_dim = np.concatenate([cc_top_dim, container_bottom_dim])
    is_on_top = c_bottom_pos_[:,np.newaxis,2] == cc_top_pos[np.newaxis,:,2]
    base_support_area = compute_collision(c_bottom_pos_[:,:2], c_bottom_dims_[:,:2], cc_top_pos[:,:2], cc_top_dim[:,:2])
    base_support_area *= is_on_top
    base_support_area = np.sum(base_support_area, axis=-1)
    base_area = dummy_dim[:,0]*dummy_dim[:,1]
    supported_base_area_ratio = base_support_area/base_area
    is_base_supported = supported_base_area_ratio>0.5
    is_feasible = np.logical_and(is_base_supported, is_not_collide)
    is_feasible = np.logical_and(is_feasible, is_not_overflow)
    return addition_points[is_feasible]


def get_addition_points(cc_positions, cc_dims, container_dim)->np.ndarray:
    if len(cc_positions)==0:
        return np.zeros([1,3], dtype=float)
    addition_points = cc_dims[:, np.newaxis,:]*np.eye(3,3)[np.newaxis,:,:] + cc_positions[:, np.newaxis, :]
    addition_points = addition_points.reshape(len(cc_positions)*3,3)
    
    addition_points = filter_infeasible_addition_points(addition_points, cc_positions, cc_dims, container_dim)
    return addition_points

def argsort_addition_points(addition_points, mode="wall-building"):
    # if mode == "layer-building"
    criteria = (addition_points[:,1], addition_points[:,0], addition_points[:,2])
    
    if mode=="wall-building":
        criteria = (addition_points[:,1], addition_points[:,2], addition_points[:,0])
    elif mode=="column-building":
        criteria = (addition_points[:,0], addition_points[:,2], addition_points[:,1])
    sorted_idx = np.lexsort(criteria)
    return sorted_idx