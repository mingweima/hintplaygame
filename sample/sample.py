import glob
import os
import pickle
import sys

sys.path.append(os.getcwd())
sys.path.append("")

import torch

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np

from game.hint_play_game import TwoRoundHintGame
from train_qlearner import obs_to_agent


def sample_games(p1, p2, episodes=10000, verbose=False):
    hp1 = p1.hp
    hp2 = p2.hp
    env = TwoRoundHintGame(hp=hp1)

    rewards = []

    #     print(f'agents are {hp1.agent_type} and {hp2.agent_type}')

    for i_episode in range(episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        if verbose:
            env.render()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp1)
        if hp1.agent_type != 'FF':
            obs1_a1 = obs1_a1.reshape(-1, hp1.nlab1 + hp1.nlab2).T
        # P1 select and perform a hint
        # obs1_a1 = torch.tensor(obs1_a1, device=device)
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device), evaluate=True)
        obs2, _, _, _ = env.step(a1)
        if verbose:
            env.render()
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp2)
        if hp2.agent_type != 'FF':
            obs2_a2 = obs2_a2.reshape(-1, hp2.nlab1 + hp2.nlab2).T
        a2 = p2.select_action(torch.tensor([obs2_a2], device=device), evaluate=True)
        _, r, _, _ = env.step(a2)
        r = torch.tensor([r[0]], device=device)
        rewards.append(r.numpy()[0])
        if verbose:
            env.render()
    return np.array(rewards)


if __name__ == "__main__":
    agent1s = []
    agent2s = []
    aid = 11
    for filename in sorted(glob.glob(os.path.join("res/Att3_hs_5_l1_3_l2_3_TrueTrue3000000", "*.pkl"))):
        with open(filename, "rb") as f:
            res = pickle.load(f)
            agent1s += [res['p1']]
            agent2s += [res['p2']]
    print(sample_games(agent1s[aid], agent2s[aid], episodes=10, verbose=True))
