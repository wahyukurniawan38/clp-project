import numpy as np
import pandas as pd

def selected_indexes (X: np.ndarray) ->np.ndarray:
    selected_indices = np.where(X == 1)
    new_indices = np.argsort(selected_indices[0])
    selected_indexes = selected_indices[1][new_indices]
    
    return selected_indexes

def obj_1(x: np.ndarray, 
          df_cargos:pd.DataFrame, 
          df_containers:pd.DataFrame) -> float:
    index = selected_indexes(x)
    item_prices = np.array(df_cargos['price'])
    item_volumes = np.array(df_cargos['vol'])
    cost = np.array(df_containers['e'])

    selected_volumes = item_volumes[index]
    selected_prices = item_prices[index]
    total_revenue = np.sum(selected_volumes * selected_prices)
    total_cost = np.sum(cost[np.any(x, axis=1)])
    total_revenue_all = np.sum(item_volumes * item_prices)    
    return (total_revenue - total_cost) / total_revenue_all

def obj_2(x: np.ndarray, 
          df_cargos:pd.DataFrame, 
          df_containers:pd.DataFrame) -> float:
    is_container_used = np.any(x>0, axis =1)
    container_volumes = np.array(df_containers['volume'])
    item_volumes = np.array(df_cargos['vol'])
    cargo_vol_percont = x.dot(item_volumes)
    used_container_volumes = container_volumes[is_container_used]
    volume_per_container = cargo_vol_percont[is_container_used]
    residual_space = container_volumes - volume_per_container
    max_residual = np.max(residual_space)
    min_residual = np.min(residual_space)
    max_used_volume = used_container_volumes.max()
    obj_value = (max_residual - min_residual) / max_used_volume
    return obj_value

def compute_obj(x:np.ndarray, 
                df_cargos:pd.DataFrame, 
                df_containers:pd.DataFrame,
                omega:float=0.99):
    y1 = obj_1(x,df_cargos,df_containers)
    y2 = obj_2(x,df_cargos,df_containers)
    return omega*y1-(1-omega)*y2