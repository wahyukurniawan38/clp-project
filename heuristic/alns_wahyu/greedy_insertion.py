import numpy as np
import pandas as pd
"""
    cargo_container_mask: np array with dim [num_container, num_cargo]
    df_cargos: items datafram
    df_containers
"""
def greedy_insertion(cargo_container_mask:np.ndarray, 
                     df_cargos:pd.DataFrame,
                     df_containers:pd.DataFrame)->np.ndarray:
    ratio = np.array(df_cargos['ratio'])
    item_weights = np.array(df_cargos['weight'])
    item_volumes = np.array(df_cargos['vol'])
    container_volumes = np.array(df_containers['volume'])
    container_weights = np.array(df_containers['weight'])

    unplaced_item_indices = np.where(~np.any(cargo_container_mask, axis=0))[0]
    sorted_unplaced_indices = unplaced_item_indices[np.argsort(-ratio[unplaced_item_indices])]
    used_weights = cargo_container_mask.dot(item_weights)
    used_volumes = cargo_container_mask.dot(item_volumes)
    for item_idx in sorted_unplaced_indices:
        can_fit = (item_weights[item_idx] <= (container_weights - used_weights)) & \
                  (item_volumes[item_idx] <= (container_volumes - used_volumes))
                  
        eligible_containers = np.where(can_fit)[0]
        
        if eligible_containers.size > 0:
            # Pilih container dengan muatan volume terendah yang sudah memiliki item, jika memungkinkan
            non_empty_containers = [c for c in eligible_containers if used_volumes[c] > 0]
            if non_empty_containers:
                chosen_container = non_empty_containers[np.argmin(used_volumes[non_empty_containers])]
            else:
                # Jika semua container yang memenuhi syarat kosong, pilih yang memiliki muatan volume terendah
                chosen_container = eligible_containers[np.argmin(used_volumes[eligible_containers])]
            
            cargo_container_mask[chosen_container, item_idx] = 1
            used_weights[chosen_container] += item_weights[item_idx]
            used_volumes[chosen_container] += item_volumes[item_idx]
    return cargo_container_mask