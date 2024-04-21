import pathlib
import random

import matplotlib
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from heuristic.alns_wahyu.alns import ALNS_W
from heuristic.alns_wahyu.arguments import prepare_args
from heuristic.alns_wahyu.evaluator.safak_evaluator import SafakEvaluator
from heuristic.alns_wahyu.operator.destroy import RandomRemoval, WorstRemoval
from heuristic.alns_wahyu.operator.repair import GreedyRepair, RandomRepair


def setup_destroy_operators(args):
    operator_list = [RandomRemoval(), WorstRemoval()]
    return operator_list
    
def setup_repair_operators(args):
    operator_list = [RandomRepair(), GreedyRepair()]
    return operator_list

def setup_log_writer(args):
    summary_root = "logs"
    summary_dir = pathlib.Path(".")/summary_root
    experiment_summary_dir = summary_dir/args.title
    experiment_summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=experiment_summary_dir.absolute())
    return writer
    

def run(args):
    data_path = pathlib.Path()/"instances"/"data_from_wahyu"/args.instance_filename
    df_cargos = pd.read_excel(data_path.absolute(),sheet_name='Item', header=0)
    df_containers = pd.read_excel(data_path.absolute(),sheet_name='Container', header=0)
    destroy_operators = setup_destroy_operators(args)
    repair_operators = setup_repair_operators(args)
    log_writer = setup_log_writer(args)
    alns_solver = ALNS_W(destroy_operators,
                         repair_operators,
                         SafakEvaluator(),
                         log_writer,
                         args.max_iteration,
                         args.max_feasibility_repair_iteration,
                         args.omega,
                         args.a,
                         args.b1,
                         args.b2,
                         args.d1,
                         args.d2)
    current_result, best_result = alns_solver.solve(df_cargos, df_containers)
    
        


if __name__ == "__main__":
    args = prepare_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    matplotlib.use("Agg")
    run(args)