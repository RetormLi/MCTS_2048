#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
import random


class TreeNode():
    def __init__(self, N, Q, parent):
        super().__init__()
        self.N = N
        self.Q = Q
        self.parent = parent
        self.children = dict()

    def expand(self):
        pass


class MCTS():
    def __init__(self, state, depth, c, rollout_limit):
        self.state = state
        self.depth = depth
        self.c = c
        self.root = TreeNode(N=0, Q=0, parent=None)
        self.rollout_limit = rollout_limit

    def selectAction(self):
        while True:
            simulate()

    def simulate(self):
        if self.depth == 0:
            return
        pass

    def rollout(self):
        pass

    def selection(self):
        pass

    def backProp(self):
        pass
