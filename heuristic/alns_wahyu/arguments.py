import argparse
import sys


def prepare_args():
    parser = argparse.ArgumentParser(description='ALNS-WAHYU')
    parser.add_argument('--instance-filename',
                        type=str,
                        default="data-full.xlsx",
                        help="excel filename for testing")    
    parser.add_argument('--title',
                        type=str,
                        default="experiment-default-param-1",
                        help="title for experiment, differentiate between different experiement")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='seed for random generator')
    parser.add_argument('--num-replication',
                        type=int,
                        default=10,
                        help='number of experiment replications')
    
    
    # Hyperparam starts here
    parser.add_argument('--max-iteration',
                        type=int,
                        default=10,
                        help='maximum alns iteration')
    parser.add_argument('--max-feasibility-repair-iteration',
                        type=int,
                        default=10,
                        help='maximum feas repair iteration')
    parser.add_argument('--omega',
                        type=float,
                        default=0.99,
                        help='omega for objectives weighted sum')
    parser.add_argument('--a',
                        type=float,
                        default=0.9,
                        help='the operator score update parameter')
    parser.add_argument('--b1',
                        type=float,
                        default=1.5,
                        help='the operator score update parameter when improving')
    parser.add_argument('--b2',
                        type=float,
                        default=0.6,
                        help='the operator score update parameter when not improving')     
    parser.add_argument('--d1',
                        type=float,
                        default=0.1,
                        help='to compute degree of destruction for destroy operator')
    parser.add_argument('--d2',
                        type=float,
                        default=0.7,
                        help='to compute degree of destruction for destroy operator')     
    parser.add_argument('--insertion-mode',
                        type=str,
                        choices=["wall-building", "layer-building", "column-building"],
                        default="wall-building",
                        help='constructinve heuristic insertion mode')    
    parser.add_argument('--cargo-sort-criterion',
                        type=str,
                        choices=["random","lwh,dec", "lwh,inc", "h-base-area,dec-inc", "base-area,dec","base-area,inc", "wall-area,dec", "wall-area,inc", "column-area,dec", "column-area,inc"],
                        default="random",
                        help='cargo ordering criterion for the constructive heuristic')
         
    
    
    
    args = parser.parse_args(sys.argv[1:])
    return args