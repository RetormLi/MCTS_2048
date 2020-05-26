#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
import random
import math

actions = [0, 1, 2, 3]


class TreeNode:
    def __init__(self, state, r, N, Q, parent):
        self.state = state
        self.N = N
        self.Q = Q
        self.r = r
        self.parent = parent
        self.children = dict()

    def expand(self):
        for action in actions:
            # self.N += 1
            self.children[action] = TreeNode(
                None, r=0, N=1, Q=0, parent=self)

    def set_node(self, state, r):
        self.state = state
        self.r = r

    def is_leaf(self):
        return self.children == dict()


class MCTS:
    def __init__(self, state, depth=5, c=100, rollout_limit=10, gamma=0.9, simulate_depth=5):
        self.state = state
        self.depth = depth
        self.c = c
        self.gamma = gamma
        self.simulate_depth = simulate_depth
        self.root = TreeNode(state, r=0, N=1, Q=0, parent=None)
        self.rollout_limit = rollout_limit

    def select_action(self):
        for i in range(self.depth):
            self.simulate(self.root, self.depth)

        max_q = 0
        best_action = 0
        for action in actions:
            new_node = self.root.children[action]
            value = new_node.Q
            if value > max_q:
                max_q = value
                best_action = action
        return best_action

    def simulate(self, node, d):
        """
        Simulation step for MCTS
        """
        if d == 0:
            return 0
        if node.is_leaf():
            node.expand()
            return self.rollout(node.state, self.rollout_limit)
        action = self.selection(node)

        new_state = copy.deepcopy(self.state)
        reward = new_state.step(action)[1]
        new_node = node.children[action]
        new_node.set_node(new_state, reward)

        q = reward + self.gamma * self.simulate(new_node, d - 1)
        new_node.N += 1
        node.N += 1
        new_node.Q += (q - new_node.Q) / new_node.N

        return q

    def selection(self, node):
        max_q = 0
        best_action = 0
        for action in actions:
            new_node = node.children[action]
            value = new_node.Q + self.c * \
                    math.sqrt(math.log(node.N) / new_node.N)
            if value > max_q:
                max_q = value
                best_action = action
        return best_action

    def rollout(self, env, limit):
        if limit == 0:
            return 0
        action = random.choice(actions)
        reward = env.step(action)[1]
        return reward + self.gamma * self.rollout(env, limit - 1)
