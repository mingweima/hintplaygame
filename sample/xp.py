import glob
import os
import pickle

import torch

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
from pandas import DataFrame

from game.hint_play_game import TwoRoundHintGame
from train_qlearner import obs_to_agent

from game.hyperparams import Hp
from agent.qlearner import QLearner


def sample_games(p1, p2, episodes=10000, verbose=False):
    hp1 = p1.hp
    hp2 = p2.hp
    env = TwoRoundHintGame(hp=hp1)

    rewards = []

    print(f'agents are {hp1.agent_type} and {hp2.agent_type}')

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


def sp_test(agent_path, verb=False):
    agent1s = []
    agent2s = []
    for filename in glob.glob(os.path.join(agent_path, "*.pkl")):
        with open(filename, "rb") as f:
            res = pickle.load(f)
            agent1s += [res['p1']]
            agent2s += [res['p2']]
    score_dict = {}
    for idx1, p1 in enumerate(agent1s):
        score_dict[idx1] = {}
        for idx2, p2 in enumerate(agent2s):
            print(idx1, idx2, )
            score_dict[idx1][idx2] = sample_games(p1, p2, episodes=1000, verbose=verb).mean()

    return DataFrame(score_dict)


def xp_test(agent_path, model='ff', verb=False):
    agent1s = []
    agent2s = []
    for filename in glob.glob(os.path.join(agent_path, "*.pkl")):
        with open(filename, "rb") as f:
            res = pickle.load(f)
            agent1s += [res['p1']]
            agent2s += [res['p2']]
    score_dict = {}
    for idx1, p1 in enumerate(agent1s):
        score_dict[idx1] = {}
        for idx2, p2 in enumerate(agent2s):
            score_dict[idx1][idx2] = sample_games(p1, p2, episodes=1000, verbose=verb).mean()
            print(idx1, idx2, score_dict[idx1][idx2])
    return DataFrame(score_dict)


if __name__ == "__main__":
    # print('test start')
    # score_df = sp_test(
    #     '/Users/liujizhou/Desktop/ReinforcementLearning/some work/hintplaygame/res/teach/FF_hand_5_l1_3_l2_3_Att3_hand_5_l1_3_l2_3',
    #     model='att')
    # print(score_df)
    # score_df.to_csv('xp_teach_att3_ff.csv')

    print('test start')
    score_df = sp_test('/Users/mmw/Documents/GitHub/hintplaygame/res//FF_hs_3_l1_3_l2_3_TrueTrue2000000', verb=True)
    print(score_df)
    # score_df.to_csv('xp_att2.csv')

    # print('test start')
    # score_df = sp_test('res/att3', model='att', verb=False)
    # print(score_df)
    # score_df.to_csv('xp_att3.csv')

    # print('test start')
    # score_df = sp_test('res/Att3_hand_5_l1_4_l2_4', model='att', verb=False)
    # print(score_df)
    # score_df.to_csv('xp_att3_l_4.csv')

    # agent1s = []
    # agent2s = []
    # for filename in glob.glob(os.path.join("res/att3", "*.pkl")):
    #   with open(filename, "rb") as f:
    #       res = pickle.load(f)
    #       agent1s += [res['p1']]
    #       agent2s += [res['p2']]
    # print(sample_games_att(agent1s[1], agent2s[1], episodes=10, verbose=True))

    # print(mechanical_test(verb=False))
