import numpy as np
import pandas as pd

def selected_indexes (X: np.ndarray) ->np.ndarray:
    selected_indices = np.where(X == 1)
    new_indices = np.argsort(selected_indices[0])
    selected_indexes = selected_indices[1][new_indices]
    
    return selected_indexes

def obj_1(x: np.ndarray, 
          cargo_volumes: np.ndarray,
          cargo_prices: np.ndarray,
          container_cost: float) -> float:
    index = selected_indexes(x)
    selected_volumes = cargo_volumes[index]
    selected_prices = cargo_prices[index]
    total_revenue = np.sum(selected_volumes * selected_prices)
    is_container_used = np.any(x, axis=1)
    total_cost = np.sum(container_cost*is_container_used)
    total_revenue_all = np.sum(cargo_volumes * cargo_prices)    
    return (total_revenue - total_cost) / total_revenue_all

def obj_2(x: np.ndarray, 
          cargo_volumes:np.ndarray,
          container_volume:float) -> float:
    is_container_used = np.any(x>0, axis =1)
    cargo_vol_percont = x.dot(cargo_volumes)
    used_container_volumes = np.sum(is_container_used*container_volume)
    volume_per_container = cargo_vol_percont[is_container_used]
    residual_space = container_volume - volume_per_container
    max_residual = np.max(residual_space)
    min_residual = np.min(residual_space)
    max_used_volume = used_container_volumes.max()
    obj_value = (max_residual - min_residual) / max_used_volume
    return obj_value

def compute_obj(x:np.ndarray, 
                cargo_volumes: np.ndarray,
                cargo_prices: np.ndarray,
                container_cost: float,
                container_volume: float,
                omega:float=0.99):
    y1 = obj_1(x, cargo_volumes, cargo_prices, container_cost)
    y2 = obj_2(x, cargo_volumes, container_volume)
    return omega*y1-(1-omega)*y2