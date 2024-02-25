from solver.problem import Problem
from solver.solution import SolutionBase

class Solution(SolutionBase):
    def __init__(self, problem: Problem, **kwargs):
        super().__init__(problem, **kwargs)
        self.block_dict = kwargs.get("block_dict") or {}
        self.blocks = kwargs.get("blocks") or []
        self.new_block_dict = kwargs.get("new_block_dict") or {}
        self.new_blocks = kwargs.get("new_blocks") or []