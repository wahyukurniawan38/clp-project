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
  +------+ 
 /|     /| 
+-+----+ | <-It's this side
| |    | | the surface that moves along the x-axis
| +----+-+ for the x-axis projection
|/     |/  
+------+   
"""
def get_x_surfaces(container_dim:np.ndarray,
                    cc_dims:np.ndarray,
                    cc_positions:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    x_s0 = cc_positions + cc_dims*np.asanyarray([[1,0,0]])
    x_s1 = cc_positions + cc_dims
    x_s0 = np.append(x_s0, np.zeros([1,3]),axis=0)
    x_s1 = np.append(x_s1, [container_dim-container_dim*np.asanyarray([1,0,0])], axis=0)
    return x_s0, x_s1

"""  
     | it's this surface,
     v that moves along the y-axis
  +------+  it is NOT the top lid
 /|     /| 
+-+----+ | 
| |    | | 
| +----+-+ 
|/     |/  
+------+   
"""
def get_y_surfaces(container_dim:np.ndarray,
                    cc_dims:np.ndarray,
                    cc_positions:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    y_s0 = cc_positions + cc_dims*np.asanyarray([[0,1,0]])
    y_s1 = cc_positions + cc_dims
    y_s0 = np.append(y_s0, np.zeros([1,3]),axis=0)
    y_s1 = np.append(y_s1, [container_dim-container_dim*np.asanyarray([0,1,0])], axis=0)
    return y_s0, y_s1

"""  
     | it's the TOP LID now,
     v that moves along the z-axis
  +------+  
 /|     /| 
+-+----+ | 
| |    | | 
| +----+-+ 
|/     |/  
+------+   
"""
def get_z_surfaces(container_dim:np.ndarray,
                    cc_dims:np.ndarray,
                    cc_positions:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    z_s0 = cc_positions + cc_dims*np.asanyarray([[0,0,1]])
    z_s1 = cc_positions + cc_dims
    z_s0 = np.append(z_s0, np.zeros([1,3]),axis=0)
    z_s1 = np.append(z_s1, [container_dim-container_dim*np.asanyarray([0,0,1])], axis=0)
    return z_s0, z_s1

"""
    In the original, support area for insertion point is not considered
    I think this limits the possible solution.
    Support area only checked in insertion feasibility, not here.

    p (3,)
    container_dim (3,)
    cc_dims (N,3)
    cc_positions (N,3)
    axis=0 means x-axis, 1=y-axis, 2=z-axis
"""      
def project_extreme_point(p: np.ndarray,
                          container_dim:np.ndarray,
                          cc_dims:np.ndarray,
                          cc_positions:np.ndarray,
                          axis:int=0):
    if axis==0:
        p0 = p - 999999*np.asanyarray([1,0,0])
        x_s0, x_s1 = get_x_surfaces(container_dim, cc_dims, cc_positions)
        is_collide = check_collision_3d_vectorized(p0, p, x_s0, x_s1)
        collide_point = x_s1[is_collide,0]
        p[0] = np.max(collide_point)
        return p
    if axis==1:
        p0 = p - 999999*np.asanyarray([0,1,0])
        y_s0, y_s1 = get_y_surfaces(container_dim, cc_dims, cc_positions)
        is_collide = check_collision_3d_vectorized(p0, p, y_s0, y_s1)
        collide_point = y_s1[is_collide,1]
        p[1] = np.max(collide_point)
        return p
    # # if axis==2
    p0 = p - 999999*np.asanyarray([0,0,1])
    z_s0, z_s1 = get_z_surfaces(container_dim, cc_dims, cc_positions)
    is_collide = check_collision_3d_vectorized(p0, p, z_s0, z_s1)
    collide_point = z_s1[is_collide,2]
    p[2] = np.max(collide_point)
    return p

"""
    cc_ prefix means cargo already inside the container
    c_ prefix means cargo to be inserted
"""
def find_ip_and_cargo_idx(cc_dims:np.ndarray, 
                          cc_positions:np.ndarray, 
                          insertion_points:np.ndarray, 
                          c_dims:np.ndarray, 
                          c_weights:np.ndarray, 
                          c_volumes:np.ndarray, 
                          cargo_sort:str="volume-height", 
                          ip_sort:str="first-fit"):
    ip_idx, chosen_cargo_idx = None, None #default value
    print(possible_rotation_mats.shape)
    if cargo_sort == "volume-height":
        c_heights = c_dims[:, 2]
        print(c_heights)
        print(c_volumes)
    exit()
        