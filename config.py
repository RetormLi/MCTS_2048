import random


class Config:
    """
    The config of MCTS.
    """
    def __init__(self):
        self.c = 100
        self.gamma = 0.7
        self.simulation_depth = 20
        self.rollout_depth = 20
        self.iterations = 50
        self.rollout_policy = random.choice
        # self.c = 50
        # self.gamma = 0.4
        # self.simulation_depth = 10
        # self.rollout_depth = 10
        # self.iterations = 100
        # self.rollout_policy = random.choice
