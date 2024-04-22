import math
from typing import Tuple

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
from tqdm import tqdm

from heuristic.alns_wahyu.evaluator.evaluation_result import create_copy
from heuristic.alns_wahyu.evaluator.evaluator import Evaluator
from heuristic.alns_wahyu.evaluator.evaluation_result import EvaluationResult, is_better
from heuristic.alns_wahyu.initialization import initialize_x
from heuristic.alns_wahyu.objective import compute_obj
from heuristic.alns_wahyu.operator.feasibility_repair import feasibility_repair


class ALNS_W():
    def __init__(self,
                 destroy_operators,
                 repair_operators,
                 evaluator:Evaluator,
                 log_writer: SummaryWriter,
                 max_iteration:int=10,
                 max_feasibility_repair_iteration:int=10,
                 omega:float=0.99,
                 a:float=0.9,
                 b1:float=1.5,
                 b2:float=0.6,
                 d1:float=0.1,
                 d2:float=0.7) -> None:
        self.evaluator:Evaluator = evaluator
        self.destroy_operators = destroy_operators
        self.repair_operators = repair_operators
        self.log_writer = log_writer
        self.max_iteration = max_iteration
        self.max_feasibility_repair_iteration = max_feasibility_repair_iteration
        self.omega = omega
        self.a = a
        self.b1 = b1
        self.b2 = b2
        self.d1 = d1
        self.d2 = d2
        
    def log(self, 
            eval_result:EvaluationResult, 
            best_eval_result:EvaluationResult, 
            destroy_operators,
            destroy_counts,
            destroy_scores,
            repair_operators,
            repair_counts,
            repair_scores,
            t:int):
        self.log_writer.add_scalar("Score", eval_result.score, t)
        self.log_writer.add_scalar("Best Score", best_eval_result.score, t)
        self.log_writer.add_scalar("Volume Packed (ratio)", eval_result.volume_packed_ratio, t)
        self.log_writer.add_scalar("Weight Packed (ratio)", eval_result.weight_packed_ratio, t)
        self.log_writer.add_scalar("Feasibility_COG", eval_result.cog_feasibility_ratio, t)
        self.log_writer.add_scalar("Feasibility_Packing", eval_result.packing_feasibility_ratio, t) 
        self.log_writer.add_scalar("Num Container Used", np.sum(np.any(eval_result.x, axis=1)), t) 
        for d_idx in range(len(destroy_operators)):
            self.log_writer.add_scalar(str(destroy_operators[d_idx])+" Count", destroy_counts[d_idx], t)
            self.log_writer.add_scalar(str(destroy_operators[d_idx])+" Score", destroy_scores[d_idx], t) 
        for r_idx in range(len(repair_operators)):
            self.log_writer.add_scalar(str(repair_operators[r_idx])+" Count", repair_counts[r_idx], t)
            self.log_writer.add_scalar(str(repair_operators[r_idx])+" Score", repair_scores[r_idx], t) 
        
        # log the container figure
        for ct_idx in range(len(eval_result.solution_list)):
            fig = eval_result.get_container_fig(ct_idx)
            if fig is not None:
                self.log_writer.add_figure("Container Fig-"+str(ct_idx), fig, t)
            
        
    def solve(self,
              df_cargos:pd.DataFrame,
              df_containers:pd.DataFrame) -> Tuple[EvaluationResult,EvaluationResult]:
        print("Initializing x")
        x = initialize_x(df_cargos, df_containers)
        print("Initializing x completed")
        print("Begin initial evaluation")
        eval_result = self.evaluator.evaluate(x, df_cargos,  df_containers, self.omega)
        print("Completed initial evaluation")
        cargo_volumes = eval_result.cargo_volumes
        cargo_prices = eval_result.cargo_prices
        cargo_weights = eval_result.cargo_weights
        cargo_ratios = eval_result.cargo_ratios
        cargo_loads = eval_result.cargo_loads
        container_max_volume = eval_result.max_container_volume
        container_max_weight = eval_result.max_container_weight
        container_cost = eval_result.container_cost
        
        score = eval_result.score

        destroy_scores = np.ones((len(self.destroy_operators),))
        repair_scores = np.ones((len(self.repair_operators),))
        destroy_counts = np.zeros((len(self.destroy_operators),))
        repair_counts= np.zeros((len(self.repair_operators),))

        best_x = x.copy()
        best_eval_result = create_copy(eval_result)
        best_score = best_eval_result.score
        self.log(eval_result, 
                 best_eval_result, 
                 self.destroy_operators,
                 destroy_counts,
                 destroy_scores,
                 self.repair_operators,
                 repair_counts,
                 repair_scores,
                 t=0)
        
        operator_arguments = {
            "cargo_volumes": cargo_volumes,
            "cargo_weights": cargo_weights,
            "cargo_ratios": cargo_ratios,
            "cargo_loads": cargo_loads,
            "container_max_volume": container_max_volume,
            "container_max_weight": container_max_weight,
        }
        
        not_improving_count = 0
        patience = 10
        for t in tqdm(range(1, self.max_iteration+1), "ALNS Main Iteration"):
            # preparing (destroy) operator arguments
            dest_degree = (self.d2-self.d1)/t + self.d1
            operator_arguments["nof"] = math.ceil(dest_degree*len(np.flatnonzero(x)))
            operator_arguments["nof2"] = round(dest_degree*len(x))
            p_destroy = destroy_scores/destroy_scores.sum()
            p_repair = repair_scores/repair_scores.sum()
            destroy_op_idx = np.random.choice(len(self.destroy_operators), p=p_destroy)
            destroy_op = self.destroy_operators[destroy_op_idx]
            destroy_counts[destroy_op_idx]+=1
            repair_op_idx = np.random.choice(len(self.repair_operators), p=p_repair)
            repair_op = self.repair_operators[repair_op_idx]
            repair_counts[repair_op_idx]+=1
            next_x = x.copy()
            next_x = destroy_op(next_x, **operator_arguments)
            next_x = repair_op(next_x, **operator_arguments)
            next_score = compute_obj(next_x,
                                     cargo_volumes,
                                     cargo_prices,
                                     container_cost,
                                     container_max_volume,
                                     self.omega)
            if not eval_result.is_feasible or (eval_result.is_feasible and score < next_score):
                next_eval_result = self.evaluator.evaluate(next_x, df_cargos, df_containers, self.omega)
                if not next_eval_result.is_feasible:
                    next_eval_result = feasibility_repair(next_eval_result, self.evaluator, self.max_feasibility_repair_iteration)
                next_score = next_eval_result.score
                
                if is_better(next_eval_result, eval_result):
                    destroy_scores[destroy_op_idx] = self.a*destroy_scores[destroy_op_idx]+(1-self.a)*self.b1
                    repair_scores[repair_op_idx] = self.a*repair_scores[repair_op_idx]+(1-self.a)*self.b1 
                    not_improving_count = 0
                    # x = next_x.copy()
                    # eval_result = next_eval_result
                    # score = next_score   
                else:
                    # revert solution if straying too far not improving
                    not_improving_count += 1
                    destroy_scores[destroy_op_idx] = self.a*destroy_scores[destroy_op_idx]+(1-self.a)*self.b2
                    repair_scores[repair_op_idx] = self.a*repair_scores[repair_op_idx]+(1-self.a)*self.b2
                
                
                # update solution
                x = next_x.copy()
                eval_result = next_eval_result
                score = next_score   
                
                # print(next_eval_result.is_all_cog_feasible, best_eval_result.is_all_cog_feasible)
                # print(next_eval_result.cog_feasibility_ratio, best_eval_result.cog_feasibility_ratio)
                # print(next_eval_result.is_all_cargo_packed, best_eval_result.is_all_cargo_packed)
                # print(next_eval_result.packing_feasibility_ratio, best_eval_result.packing_feasibility_ratio)
                if is_better(next_eval_result, best_eval_result):
                    # print("yes")
                    # print(next_eval_result.cog_feasibility_ratio)
                    # print
                    best_x = next_x.copy()
                    best_eval_result = create_copy(next_eval_result)
                    best_score = next_score        
                self.log(eval_result, 
                         best_eval_result, 
                         self.destroy_operators,
                         destroy_counts,
                         destroy_scores,
                         self.repair_operators,
                         repair_counts,
                         repair_scores,
                         t)
            else:
                not_improving_count += 1
            if not_improving_count == patience:
                x = best_x.copy()
                eval_result = create_copy(best_eval_result)
                score = best_score
    
        return eval_result, best_eval_result
    