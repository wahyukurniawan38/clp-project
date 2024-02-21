import numpy as np

from heuristic.lns_safak.solution import Solution
from solver.problem import Problem
from heuristic.lns_safak.constructive import constructive_heuristic
from heuristic.utils import add_container

# add container

# def lns(problem: Problem, 
#         d
#         insertion_mode="layer-building") -> Solution:
#     solution = Solution(problem)
#     for i in range(len(problem.container_type_list)):
#         for j in range(problem.container_type_list[i].num_available):
#             solution = add_container(solution, i)
#     solution = constructive_heuristic(solution)