from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from solver.problem import Problem

def init_positions(problem:Problem, positions: np.ndarray=None):
    if positions is not None:
        return positions.copy()
    total_num_cargo = sum([cargo_type.num_cargo for cargo_type in problem.cargo_type_list])
    positions = -np.ones([total_num_cargo, 3], dtype=float)
    return positions

def init_cargo_container_maps(problem: Problem, cargo_container_maps:np.ndarray=None):
    if cargo_container_maps is not None:
        return cargo_container_maps.copy()
    total_num_cargo = sum([cargo_type.num_cargo for cargo_type in problem.cargo_type_list])
    cargo_container_maps = -np.ones([total_num_cargo,], dtype=float)
    return cargo_container_maps

def init_rotation_mats(problem: Problem, rotation_mats: np.ndarray=None):
    if rotation_mats is not None:
        return rotation_mats.copy()
    total_num_cargo = sum([cargo_type.num_cargo for cargo_type in problem.cargo_type_list])
    rotation_mats = np.eye(3,3,dtype=int)[np.newaxis,:,:]
    rotation_mats = np.repeat(rotation_mats, total_num_cargo, axis=0)
    return rotation_mats

def init_nums_container_used(problem: Problem, nums_container_used: List[int]=None):
    if nums_container_used is not None:
        return deepcopy(nums_container_used)
    
    return [0]*len(problem.container_type_list)
    
def init_container_dims(container_dims:np.ndarray=None):
    if container_dims is not None:
        return container_dims.copy()
    return np.empty([0,3],dtype=float)

def init_container_max_volumes(container_max_volumes:np.ndarray=None):
    if container_max_volumes is not None:
        return container_max_volumes.copy()
    return np.empty([0,],dtype=float)

def init_container_filled_volumes(container_filled_volumes:np.ndarray=None):
    if container_filled_volumes is not None:
        return container_filled_volumes.copy()
    return np.empty([0,],dtype=float)

def init_container_max_weights(container_max_weights:np.ndarray=None):
    if container_max_weights is not None:
        return container_max_weights.copy()
    return np.empty([0,],dtype=float)

def init_container_filled_weights(container_filled_weights:np.ndarray=None):
    if container_filled_weights is not None:
        return container_filled_weights.copy()
    return np.empty([0,],dtype=float)

def get_possible_rotation_mats():
    eye = np.eye(3,3)
    row_permutations = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i==j or i==k or k==j:
                    continue
                row_permutations += [[i,k,j]]

    possible_rotation_mats = [eye[[r],:] for r in row_permutations]
    possible_rotation_mats = np.concatenate(possible_rotation_mats, axis=0)
    return possible_rotation_mats

def plot_cube(ax, x, y, z, dx, dy, dz, color='red', text_annot:str=""):
    """ Auxiliary function to plot a cube. code taken somewhere from the web.  """
    xx = [x, x, x+dx, x+dx, x]
    yy = [y, y+dy, y+dy, y, y]
    kwargs = {'alpha': 1, 'color': color}
    artists = []
    artists+= ax.plot3D(xx, yy, [z]*5, **kwargs)
    artists+= ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
    artists+= ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
    artists+= ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
    artists+= ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
    artists+= ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
    if text_annot!="":
        artists+= [ax.text(x+dx/2,y,z+dz/2,text_annot,None,fontweight=1000)]
    return artists

def visualize_box(container_dim:np.ndarray,
                  cc_positions:np.ndarray,
                  cc_dims:np.ndarray,
                  cc_rotation_mats:np.ndarray):
    
    fig = plt.figure()
    axGlob = fig.add_subplot(projection='3d')
    # . plot scatola 
    plot_cube(axGlob,0, 0, 0, float(container_dim[0]), float(container_dim[1]), float(container_dim[2]))
    # . plot intems in the box 
    colorList = ["black", "blue", "magenta", "orange"]
    counter = 0
    cc_real_dims = (cc_dims[:,np.newaxis,:]*cc_rotation_mats).sum(axis=-1)
    for i in range(len(cc_dims)):
        x,y,z = cc_positions[i,0],cc_positions[i,1],cc_positions[i,2]
        color = colorList[counter % len(colorList)]
        plot_cube(axGlob, float(x), float(y), float(z), 
                    float(cc_real_dims[i,0]), float(cc_real_dims[i,1]), float(cc_real_dims[i,2]),
                    color=color,text_annot=str(i))
        counter = counter + 1  
    plt.show() 

