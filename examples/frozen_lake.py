# -*- coding: utf-8 -*-

import gymnasium as gym
import pygame
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv


class FrozenLake:
    def __init__(self):
        self.env = gym.make('FrozenLake8x8-v1', render_mode=None)


if __name__ == "__main__":
    frozen_lake = FrozenLake()

    # VI/PI
    # V, V_track, pi = Planner(frozen_lake.env.P).value_iteration()
    # V, V_track, pi = Planner(frozen_lake.env.P).policy_iteration()

    # Q-learning
    Q, V, pi, Q_track, pi_track = RL(
        frozen_lake.env,
        10_000,
        frozen_lake.env.observation_space.n,
        frozen_lake.env.action_space.n
    ).q_learning()

    test_scores = TestEnv.test_env(env=frozen_lake.env, render=False, user_input=False, pi=pi)
