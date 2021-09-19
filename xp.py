import glob
import os
import pickle

import torch

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
from pandas import DataFrame

from hint_play_game import TwoRoundHintGame
from train_qlearner import obs_to_agent

from hyperparams import Hp
from qlearner import QLearner


def sample_games_ff(p1, p2, episodes=10000, verbose=False):
    hp = p1.hp
    env = TwoRoundHintGame(hp=hp)

    rewards = []

    for i_episode in range(episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        if verbose:
            env.render()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp)
        # P1 select and perform a hint
        # obs1_a1 = torch.tensor(obs1_a1, device=device)
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device), evaluate=True)
        obs2, _, _, _ = env.step(a1)
        if verbose:
            env.render()
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp)
        a2 = p2.select_action(torch.tensor([obs2_a2], device=device), evaluate=True)
        _, r, _, _ = env.step(a2)
        r = torch.tensor([r[0]], device=device)
        rewards.append(r.numpy()[0])
        if verbose:
            env.render()
    return np.array(rewards)


def sample_games_lstm(p1, p2, episodes=10000, verbose=False):
    hp = p1.hp
    env = TwoRoundHintGame(hp=hp)

    rewards = []

    for i_episode in range(episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        if verbose:
            env.render()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp)
        obs1_a1 = obs1_a1.reshape(-1, hp.nlab1 + hp.nlab2).T
        # P1 select and perform a hint
        # obs1_a1 = torch.tensor(obs1_a1, device=device)
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device), evaluate=True)
        obs2, _, _, _ = env.step(a1)
        if verbose:
            env.render()
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp)
        obs2_a2 = obs2_a2.reshape(-1, hp.nlab1 + hp.nlab2).T
        a2 = p2.select_action(torch.tensor([obs2_a2], device=device), evaluate=True)
        _, r, _, _ = env.step(a2)
        r = torch.tensor([r[0]], device=device)
        rewards.append(r.numpy()[0])
        if verbose:
            env.render()
    return np.array(rewards)


def sample_games_att(p1, p2, episodes=10000, verbose=False):
    hp = p1.hp
    env = TwoRoundHintGame(hp=hp)

    rewards = []

    for i_episode in range(episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        if verbose:
            env.render()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp)
        obs1_a1 = obs1_a1.reshape(-1, hp.nlab1 + hp.nlab2).T
        # P1 select and perform a hint
        # obs1_a1 = torch.tensor(obs1_a1, device=device)
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device), evaluate=True)
        obs2, _, _, _ = env.step(a1)
        if verbose:
            env.render()
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp)
        obs2_a2 = obs2_a2.reshape(-1, hp.nlab1 + hp.nlab2).T
        a2 = p2.select_action(torch.tensor([obs2_a2], device=device), evaluate=True)
        _, r, _, _ = env.step(a2)
        r = torch.tensor([r[0]], device=device)
        rewards.append(r.numpy()[0])
        if verbose:
            env.render()
    return np.array(rewards)


def sample_games_mechanical(p1, p2, episodes=10000, verbose=False):
    hp = p1.hp
    env = TwoRoundHintGame(hp=hp)

    rewards = []

    for i_episode in range(episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        if verbose:
            env.render()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp)
        obs1_a1 = obs1_a1.reshape(-1, hp.nlab1 + hp.nlab2).T
        # P1 select and perform a hint
        # obs1_a1 = torch.tensor(obs1_a1, device=device)
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device), evaluate=True)
        obs2, _, _, _ = env.step(a1)
        if verbose:
            env.render()
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp)
        obs2_a2 = obs2_a2.reshape(-1, hp.nlab1 + hp.nlab2).T
        a2 = p2.select_action(torch.tensor([obs2_a2], device=device), evaluate=True)
        _, r, _, _ = env.step(a2)
        r = torch.tensor([r[0]], device=device)
        rewards.append(r.numpy()[0])
        if verbose:
            env.render()
    return np.array(rewards)


def sp_test(agent_path, model='ff', verb=False):
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
            if model == 'ff':
                score_dict[idx1][idx2] = sample_games_ff(p1, p2, episodes=1000, verbose=verb).mean()
            if model == 'lstm':
                score_dict[idx1][idx2] = sample_games_lstm(p1, p2, episodes=1000, verbose=verb).mean()
            elif model == 'lat':
                score_dict[idx1][idx2] = sample_games_ff(p1, p2, episodes=1000, verbose=verb).mean()
            elif model == 'att':
                score_dict[idx1][idx2] = sample_games_att(p1, p2, episodes=1000, verbose=verb).mean()

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
            # if idx1 != idx2:
            if model == 'ff':
                score_dict[idx1][idx2] = sample_games_ff(p1, p2, episodes=1000, verbose=verb).mean()
            if model == 'lat':
                score_dict[idx1][idx2] = sample_games_ff(p1, p2, episodes=1000, verbose=verb).mean()
            if model == 'att':
                score_dict[idx1][idx2] = sample_games_att(p1, p2, episodes=1000, verbose=verb).mean()
            if model == 'lstm':
                score_dict[idx1][idx2] = sample_games_lstm(p1, p2, episodes=1000, verbose=verb).mean()
            print(idx1, idx2, score_dict[idx1][idx2])
    return DataFrame(score_dict)


def mechanical_test(verb=False):
    hp_train = Hp(hand_size=5,
                  nlab1=4,
                  nlab2=4,
                  shuffle_cards=False,
                  opt='adam',
                  nepsidoes=500000,
                  batch_size=512,
                  eps_scheme={'eps_start': 0.95, 'eps_end': 0.01, 'eps_decay': 50000},
                  replay_capacity=200000,
                  update_frequency=100,
                  )
    env = TwoRoundHintGame(hp=hp_train)

    p1 = QLearner(1, env, policy_type='mechanical', hp=hp_train)
    p2 = QLearner(1, env, policy_type='mechanical', hp=hp_train)
    score = sample_games_mechanical(p1, p2, episodes=1000, verbose=verb).mean()
    return score


if __name__ == "__main__":
    print('test start')
    score_df = sp_test('res/Att1_hand_5_l1_3_l2_3', model='att')
    print(score_df)
    score_df.to_csv('xp_att1.csv')

    # print('test start')
    # score_df = sp_test('res/Archive2', model='att', verb=True)
    # print(score_df)
    # # score_df.to_csv('xp_att2.csv')

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

    #print(mechanical_test(verb=False))
