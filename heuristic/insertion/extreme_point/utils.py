from typing import Tuple

import numpy as np

from solver.solution import Solution
from heuristic.utils import *

def init_insertion_points(container_dim:np.ndarray,
                          cc_dims:np.ndarray,
                          cc_positions:np.ndarray):
    num_cargos,_ = cc_dims.shape
    insertion_points = []
    for i in range(num_cargos):
        # list insertion points
        ip = cc_dims[[i],:]*np.eye(3,3)+cc_positions[[i],:]
        ip = np.repeat(ip,2,axis=0)
        ip[0,:] = project_extreme_point(ip[0,:], container_dim, cc_dims, cc_positions, axis=1)
        ip[1,:] = project_extreme_point(ip[1,:], container_dim, cc_dims, cc_positions, axis=2)
        ip[2,:] = project_extreme_point(ip[2,:], container_dim, cc_dims, cc_positions, axis=0)
        ip[3,:] = project_extreme_point(ip[3,:], container_dim, cc_dims, cc_positions, axis=2)
        ip[4,:] = project_extreme_point(ip[4,:], container_dim, cc_dims, cc_positions, axis=0)
        ip[5,:] = project_extreme_point(ip[5,:], container_dim, cc_dims, cc_positions, axis=1)
        insertion_points += [ip]
    
    insertion_points = np.unique(np.concatenate(insertion_points, axis=0),axis=0)
    return insertion_points

"""
    this is simple
    create a 3D line between the point
    and another point whose the axis to be projected
    is a negative number (any neg number)
    say the point is (1,2,3)
    projected to y axis (axis=1)
    then the line is
    (1,-1,3)(1,2,3)
    any box's right surface colliding with
    this line is compiled, then the projection
    is at the rightest surface

    point (3,)
    container_dim (3,)
    cc_dims (N,3)
    cc_positions (N,3)
    axis=0 means x-axis, 1=y-axis, 2=z-axis
"""      
def project_extreme_point(point: np.ndarray,
                          container_dim:np.ndarray,
                          cc_dims:np.ndarray,
                          cc_positions:np.ndarray,
                          axis:int=0):
    if axis==0:
        line_pos = point - 999999*np.asanyarray([1,0,0])
        line_dim = point - line_pos
        cc_front_pos, cc_front_dim = get_front_surface(cc_positions, cc_dims)
        container_back_pos = np.asanyarray([[0,0,0]])
        container_back_dim = (container_dim-container_dim*np.asanyarray([1,0,0]))[np.newaxis,:]
        cc_front_pos = np.concatenate([cc_front_pos, container_back_pos], axis=0)
        cc_front_dim = np.concatenate([cc_front_dim, container_back_dim], axis=0)
        is_collide = is_collide_3d(line_pos[np.newaxis,:], line_dim[np.newaxis, :], cc_front_pos, cc_front_dim)
        is_collide = is_collide[0,:]
        collide_point = cc_front_pos[is_collide, axis]
    if axis==1:
        line_pos = point - 999999*np.asanyarray([0,1,0])
        line_dim = point - line_pos
        cc_right_pos, cc_right_dim = get_right_surface(cc_positions, cc_dims)
        container_left_pos = np.asanyarray([[0,0,0]])
        container_left_dim = (container_dim-container_dim*np.asanyarray([0,1,0]))[np.newaxis,:]
        cc_right_pos = np.concatenate([cc_right_pos, container_left_pos], axis=0)
        cc_right_dim = np.concatenate([cc_right_dim, container_left_dim], axis=0)
        is_collide = is_collide_3d(line_pos[np.newaxis,:], line_dim[np.newaxis, :], cc_right_pos, cc_right_dim)
        is_collide = is_collide[0,:]
        collide_point = cc_right_pos[is_collide, axis]
    if axis==2:
        line_pos = point - 999999*np.asanyarray([0,0,1])
        line_dim = point - line_pos
        cc_top_pos, cc_top_dim = get_right_surface(cc_positions, cc_dims)
        container_bottom_pos = np.asanyarray([[0,0,0]])
        container_bottom_dim = (container_dim-container_dim*np.asanyarray([0,0,1]))[np.newaxis,:]
        cc_top_pos = np.concatenate([cc_top_pos, container_bottom_pos], axis=0)
        cc_top_dim = np.concatenate([cc_top_dim, container_bottom_dim], axis=0)
        is_collide = is_collide_3d(line_pos[np.newaxis,:], line_dim[np.newaxis, :], cc_top_pos, cc_top_dim)
        is_collide = is_collide[0,:]
        collide_point = cc_top_pos[is_collide, axis]
    if len(collide_point)>0:
        point[axis] = np.max(collide_point)
    return point
    

"""
    cc_ prefix means cargo already inside the container
    c_ prefix means cargo to be inserted
"""
def find_ip_and_cargo_idx(container_dim:np.ndarray,
                          cc_dims:np.ndarray, 
                          cc_positions:np.ndarray, 
                          cc_filled_weight: float,
                          cc_max_weight: float,
                          insertion_points:np.ndarray, 
                          c_dims:np.ndarray, 
                          c_weights:np.ndarray, 
                          c_volumes:np.ndarray, 
                          match_method:str="first-fit"):
    ip_idx, chosen_cargo_idx = None, None #default value
    if match_method == "first-fit":
        # i think for this method
        # feasibility checking per item
        # is more efficient
        for i in range(len(c_weights)):
            c_dim = c_dims[[i],:]
            c_weight = c_weights[[i]]
            c_volume = c_volumes[[i]]
            feasibility_mask = get_feasibility_mask(container_dim, cc_dims, cc_positions, cc_filled_weight, cc_max_weight, insertion_points, c_dim, c_weight, c_volume)
            if np.any(feasibility_mask):
                ip_idx = np.argmax(feasibility_mask[0,:])
                chosen_cargo_idx = i
                break
        return ip_idx, chosen_cargo_idx

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
    # print(is_not_collide_with_any)


    # check if the, say, i-th cargo is inserted at the j-th position
    # enough base support is provided 
    c_bottom_pos_, c_bottom_dims_ = get_bottom_surface(insertion_points_, c_dims_)
    container_bottom_pos, container_bottom_dim = get_bottom_surface(np.asanyarray([[0,0,0]]), container_dim[np.newaxis,:])
    cc_top_pos, cc_top_dim = get_top_surface(cc_positions, cc_dims)
    cc_top_pos = np.concatenate([cc_top_pos, container_bottom_pos], axis=0)
    cc_top_dim = np.concatenate([cc_top_dim, container_bottom_dim])
    is_on_top = c_bottom_pos_[:,np.newaxis,2] == cc_top_pos[np.newaxis,:,2]
    base_support_area = compute_collision(c_bottom_pos_[:,:2], c_bottom_dims_[:,:2], cc_top_pos[:,:2], cc_top_dim[:,:2])
    base_support_area *= is_on_top
    base_support_area = np.sum(base_support_area, axis=-1)
    base_area = c_dims_[:,0]*c_dims_[:,1]
    supported_base_area_ratio = base_support_area/base_area
    is_base_supported = supported_base_area_ratio>0.5
    
    # check if overflow the container
    is_not_overflow = (c_dims_ + insertion_points_) <= container_dim[np.newaxis,:]
    is_not_overflow = np.all(is_not_overflow, axis=-1)
    # combine all
    feasibility_mask = np.logical_and(is_base_supported, is_not_collide_with_any)
    feasibility_mask = np.logical_and(feasibility_mask, is_weight_cap_enough)
    feasibility_mask = np.logical_and(feasibility_mask, is_not_overflow)
    feasibility_mask = feasibility_mask.reshape([n_cargo, n_ip])
    return feasibility_mask


def argsort_cargo(c_dims:np.ndarray, 
                c_weights:np.ndarray, 
                c_volumes:np.ndarray, 
                cargo_sort:str="volume-height"):
    if cargo_sort == "volume-height":
        c_heights = c_dims[:, 2]
        sorted_idx = np.lexsort((-c_volumes, -c_heights, -c_weights),axis=0)
    return sorted_idx




        