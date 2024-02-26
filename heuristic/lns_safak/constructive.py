import numpy as np

from solver.utils import get_possible_rotation_mats
from heuristic.lns_safak.solution import Solution
from heuristic.lns_safak.insert import add_item_to_container



"""
    This is one of the two main operations
    in Safak LNS
    This operator try to pack every unpacked cargo
    into the container
    Ordering cargo is not in this method
    Ordering their rotation is in this method tho

    1. we begin by sorting the containers (instance, not type)
    2. for every container
        3. for every cargo type
            4. if all of this type is packed: continue
            5. get unpacked index
            6. reset the rotation matrix ordering for this cargo type
            7. add_item_to_container, all the unpacked items of this type
    8. check if there are no more unpacked items,
    then set is_feasible=True
"""     
def constructive_heuristic(solution:Solution, insertion_mode="layer-building"):
    container_filled_volumes = solution.container_filled_volumes
    container_costs = solution.container_costs
    sorted_container_idx = np.lexsort((-container_costs, -container_filled_volumes))
    
    c_type_sorted = np.argsort(solution.cargo_type_priority)
    for ct_idx in sorted_container_idx:
        ct_type = solution.cargo_types[ct_idx]
        for c_type in c_type_sorted:
            is_this_type = solution.cargo_types == c_type
            is_unpacked = solution.cargo_container_maps == -1
            unpacked_cargo_idx = np.nonzero(np.logical_and(is_this_type, is_unpacked))[0]
            if len(unpacked_cargo_idx)==0:
                continue
            # test for weight and volume capacity first
            volume = solution.cargo_volumes[unpacked_cargo_idx][0]
            ct_filled_volume = solution.container_filled_volumes[ct_idx]
            ct_max_volume = solution.container_max_volumes[ct_idx]
            if volume + ct_filled_volume > ct_max_volume:
                continue

            weight = solution.cargo_weights[unpacked_cargo_idx][0]
            ct_filled_weight = solution.container_filled_weights[ct_idx]
            ct_max_weight = solution.container_max_weights[ct_idx]
            if weight + ct_filled_weight > ct_max_weight:
                continue
            
            default_rotation_mat_sorted_idx = solution.default_cargo_type_rotation_sorted_idx[c_type, ct_type, :]
            solution.cargo_type_rotation_sorted_idx[c_type] = default_rotation_mat_sorted_idx
            solution, failed_to_insert_cargo_idx = add_item_to_container(solution, unpacked_cargo_idx, ct_idx, insertion_mode)
    is_packed = solution.cargo_container_maps > -1
    solution.is_feasible = np.all(is_packed)
    return solution




# is_cargo_unpacked = solution.cargo_container_maps == -1
#     unpacked_cargo_idx = np.nonzero(is_cargo_unpacked)[0]
#     unpacked_cargo_type = solution.cargo_types[unpacked_cargo_idx]
#     unpacked_cargo_priority = solution.cargo_type_priority[unpacked_cargo_type]
#     priority_sorted_idx = np.argsort(unpacked_cargo_priority)
#     unpacked_cargo_idx = unpacked_cargo_idx[priority_sorted_idx]
#     # sort container first
#     container_filled_volumes = solution.container_filled_volumes
#     container_costs = solution.container_costs
#     sorted_container_idx = np.lexsort((-container_costs, -container_filled_volumes))
#     possible_rotation_mats = get_possible_rotation_mats()
#     for container_idx in sorted_container_idx:
#         # get the default rotation matrix order
#         # for this container
#         container_dim = solution.container_dims[container_idx]
#         ct_type = solution.container_types[container_idx]
#         c_type = solution.cargo_types[unpacked_cargo_idx]
#         c_rotation_sorted_idx = solution.default_cargo_type_rotation_sorted_idx[c_type,ct_type,:]
#         c_rotation_mats = possible_rotation_mats[c_rotation_sorted_idx]
        
#         # try inserting one by one
#         for c_idx in unpacked_cargo_idx:

#         # update unpacked
#         # if inserted, update the rotation order matrix
#         # for this cargo type