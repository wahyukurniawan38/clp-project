from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import random as rand

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

def init_container_costs(container_costs=None):
    if container_costs is not None:
        return container_costs
    return  np.empty([0,], dtype=float)

def init_container_types(container_types=None):
    if container_types is not None:
        return container_types
    return  np.empty([0,], dtype=int)

def init_container_cogs(container_cogs:np.ndarray=None):
    if container_cogs is not None:
        return container_cogs.copy()
    return  np.empty([0,2], dtype=float)

def init_container_cog_tolerances(container_cog_tolerances:np.ndarray=None):
    if container_cog_tolerances is not None:
        return container_cog_tolerances
    return  np.empty([0,2], dtype=float)


def get_possible_rotation_mats():
    possible_rotation_mats = [[[1,0,0],
         [0,1,0],
         [0,0,1]],
         
         [[1,0,0],
         [0,0,1],
         [0,1,0]],
         
         [[0,1,0],
         [1,0,0],
         [0,0,1]],
         
         [[0,1,0],
         [0,0,1],
         [1,0,0]],

         [[0,0,1],
         [1,0,0],
         [0,1,0]],
         
         [[0,0,1],
         [0,1,0],
         [1,0,0]],
         ]

    possible_rotation_mats = np.asanyarray(possible_rotation_mats, dtype=float)
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
                  cc_rotation_mats:np.ndarray,
                  show=False):
    plt.close('all')
    fig = plt.figure()
    axGlob = fig.add_subplot(projection='3d')
    # . plot scatola 
    plot_cube(axGlob,0, 0, 0, float(container_dim[0]), float(container_dim[1]), float(container_dim[2]))
    # . plot intems in the box 
    colorList = ["black", "blue", "magenta", "orange"]
    counter = 0
    if cc_positions is None:
        return plt.gcf()
    cc_real_dims = (cc_dims[:,np.newaxis,:]*cc_rotation_mats).sum(axis=-1)
    for i in range(len(cc_dims)):
        x,y,z = cc_positions[i,0],cc_positions[i,1],cc_positions[i,2]
        color = colorList[counter % len(colorList)]
        plot_cube(axGlob, float(x), float(y), float(z), 
                    float(cc_real_dims[i,0]), float(cc_real_dims[i,1]), float(cc_real_dims[i,2]),
                    color=color,text_annot="")
        counter = counter + 1  
    if show:
        plt.show() 
    return plt.gcf()



# def get_random_color():
#     return "#" + "".join([rand.choice('0123456789ABCDEF') for _ in range(6)])

# def draw_cuboid(ax, position, size, color=None, label=""):
#     """Draw a cuboid"""
#     if color is None:
#         color = get_random_color()  # Use random color if none provided
#     ox, oy, oz = position
#     l, w, h = size
#     vertices = np.array([[ox, oy, oz], [ox+l, oy, oz], [ox+l, oy+w, oz], [ox, oy+w, oz],
#                          [ox, oy, oz+h], [ox+l, oy, oz+h], [ox+l, oy+w, oz+h], [ox, oy+w, oz+h]])
#     faces = [[vertices[j] for j in [0, 1, 2, 3]], [vertices[j] for j in [4, 5, 6, 7]], 
#              [vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [2, 3, 7, 6]], 
#              [vertices[j] for j in [1, 2, 6, 5]], [vertices[j] for j in [4, 7, 3, 0]]]
#     poly = Poly3DCollection(faces, facecolors=color, alpha=0.3)
#     ax.add_collection3d(poly)
#     # Add label to the center of the top face of the cuboid
#     ax.text(ox + l/2, oy + w/2, oz + h, label, color='red', ha='center', va='bottom')

# def visualize_box(container_dim:np.ndarray, cc_positions:np.ndarray, cc_dims:np.ndarray, cc_rotation_mats:np.ndarray, show=False):
#     plt.close('all')
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     # Draw the container
#     draw_cuboid(ax, (0, 0, 0), container_dim, color='grey')  # Draw the container in grey color

#     # Draw each item in the container
#     cc_real_dims = (cc_dims[:,np.newaxis,:]*cc_rotation_mats).sum(axis=-1)
#     for i in range(len(cc_dims)):
#         x, y, z = cc_positions[i]
#         dims = cc_real_dims[i]
#         color = get_random_color()  # Get a random color for each item
#         label = f"Item {i+1}"  # Label items as Item 1, Item 2, ...
#         draw_cuboid(ax, (x, y, z), dims, color, label)

#     # Set axes labels and limits
#     max_dim = max(container_dim)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_xlim(0, max_dim)
#     ax.set_ylim(0, max_dim)
#     ax.set_zlim(0, max_dim)

#     if show:
#         plt.show()
#     return plt.gcf()

