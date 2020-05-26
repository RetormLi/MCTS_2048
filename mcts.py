#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import copy
import math
from config import Config
import game2048

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
            self.children[action] = TreeNode(
                None, r=0, N=1, Q=0, parent=self)

    def set_node(self, state, r):
        self.state = state
        self.r = r

    def is_leaf(self):
        return self.children == dict()


class MCTS:
    """
    The complementation of MCTS.
    """
    def __init__(self, state: game2048.Game2048Env, config: Config):
        self.state = state
        self.iterations = config.iterations
        self.c = config.c
        self.gamma = config.gamma
        self.simulate_depth = config.simulation_depth
        self.rollout_limit = config.rollout_depth
        self.root = TreeNode(state, r=0, N=1, Q=0, parent=None)
        self.rollout_policy = config.rollout_policy

    def select_action(self) -> int:
        """
        The main interface of MCTS. Select a best action.

        :return: best action under MCTS
        """
        # simulation loop
        for i in range(self.iterations):
            self.simulate(self.root, self.iterations)

        # action choice
        max_q = 0
        best_action = 0
        for action in actions:
            new_node = self.root.children[action]
            value = new_node.Q
            if value > max_q:
                max_q = value
                best_action = action
        return best_action

    def simulate(self, node: TreeNode, d: int) -> float:
        """
        Simulation step for MCTS.

        :param node: Root state
        :param d: simulation depth
        :return: value estimation
        """
        # recursion bottom
        if d == 0:
            return 0

        # when this state does't belongs to the MCT
        if node.is_leaf():
            node.expand()
            return self.rollout(node.state, self.rollout_limit)

        # selection
        action = self.selection(node)
        new_state = copy.deepcopy(self.state)
        reward = new_state.step(action)[1]
        new_node = node.children[action]
        new_node.set_node(new_state, reward)

        q = reward + self.gamma * self.simulate(new_node, d - 1)
        new_node.N += 1
        new_node.Q += (q - new_node.Q) / new_node.N

        return q

    def selection(self, node: TreeNode) -> int:
        """
        Select the best action with UCT.

        :param node: The node represents current state.
        :return: best action
        """
        max_q = 0
        best_action = 0
        node.N = 0

        # update the value of N(s)
        for action in actions:
            node.N += node.children[action].N

        # choose action using UCT
        for action in actions:
            new_node = node.children[action]
            value = new_node.Q + self.c * math.sqrt(math.log(node.N) / new_node.N)
            if value > max_q:
                max_q = value
                best_action = action
        return best_action

    def rollout(self, env: game2048.Game2048Env, limit: int) -> float:
        """
        Rollout policy.

        :param env: last recursion state
        :param limit: rollout depth
        :return: value estimation
        """
        if limit == 0:
            return 0
        action = self.rollout_policy(actions)
        reward = env.step(action)[1]
        return reward + self.gamma * self.rollout(env, limit - 1)
