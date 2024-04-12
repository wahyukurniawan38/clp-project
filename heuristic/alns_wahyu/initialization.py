import numpy as np
import pandas as pd


def initialize_x(df_cargos: pd.DataFrame, 
                 df_containers: pd.DataFrame)->np.ndarray:
    ratio = np.array(df_cargos['ratio'])
    item_weights = np.array(df_cargos['weight'])
    item_volumes = np.array(df_cargos['vol'])
    container_volumes = np.array(df_containers['volume'])
    container_weights = np.array(df_containers['weight'])

    num_items = len(item_volumes)
    num_containers = len(container_volumes)

    x = np.zeros((num_containers, num_items), dtype=int)

    used_volumes = np.zeros(num_containers)
    used_weights = np.zeros(num_containers)

    for item_idx in range(num_items):
        containers_tried = set()
        while len(containers_tried) < num_containers:
            # Pilih kontainer secara acak
            container_idx = np.random.randint(0, num_containers)
            if container_idx in containers_tried:
                continue  # Coba kontainer lain jika kontainer ini sudah dicoba
            
            # Cek apakah item muat dalam kontainer ini
            if (used_volumes[container_idx] + item_volumes[item_idx] <= container_volumes[container_idx] and
                used_weights[container_idx] + item_weights[item_idx] <= container_weights[container_idx]):
                # Tempatkan item ke kontainer
                x[container_idx, item_idx] = 1
                used_volumes[container_idx] += item_volumes[item_idx]
                used_weights[container_idx] += item_weights[item_idx]
                break  # Berhenti mencari kontainer untuk item ini
            else:
                # Catat kontainer ini sudah dicoba
                containers_tried.add(container_idx)

    return x