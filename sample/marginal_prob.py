import glob
import os
import pickle
import sys

import numpy as np

sys.path.append(os.getcwd())
sys.path.append("")

import torch
import pandas as pd

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from game.hint_play_game import TwoRoundHintGame
from train_qlearner import obs_to_agent


def marginal_probs(p1, p2, episodes=10000,
                   l2s=['A', 'B', 'C'],
                   l1s=['1', '2', '3']):
    hp1 = p1.hp
    hp2 = p2.hp
    env = TwoRoundHintGame(hp=hp1)

    rewards = []

    target_hint = {}
    hint_play = {}

    for l1 in l1s:
        for l2 in l2s:
            target_hint[l1 + l2] = {}
            hint_play[l1 + l2] = {}
            for l1n in l1s:
                for l2n in l2s:
                    target_hint[l1 + l2][l1n + l2n] = 0
                    hint_play[l1 + l2][l1n + l2n] = 0

    for i_episode in range(episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        target_card = env.get_card_symbol_in_round()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp1)
        if hp1.agent_type != 'FF':
            obs1_a1 = obs1_a1.reshape(-1, hp1.nlab1 + hp1.nlab2).T
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device), evaluate=True)
        obs2, _, _, _ = env.step(a1)
        hint_card = env.get_card_symbol_in_round()
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp2)
        if hp2.agent_type != 'FF':
            obs2_a2 = obs2_a2.reshape(-1, hp2.nlab1 + hp2.nlab2).T
        a2 = p2.select_action(torch.tensor([obs2_a2], device=device), evaluate=True)
        _, r, _, _ = env.step(a2)
        r = torch.tensor([r[0]], device=device)
        rewards.append(r.numpy()[0])
        play_card = env.get_card_symbol_in_round()

        target_hint[target_card][hint_card] += 1
        hint_play[hint_card][play_card] += 1

    return pd.DataFrame.from_dict(target_hint, orient='index'), pd.DataFrame.from_dict(hint_play, orient='index')


if __name__ == "__main__":
    # agent1s = []
    # agent2s = []
    # aid = 0
    # file_des = 'Att2'
    # episodes = 10000
    # filelist = sorted(glob.glob(os.path.join("res/Att2_hs_5_l1_3_l2_3_FalseFalse4000000", "*.pkl")))
    # with open(filelist[aid], "rb") as f:
    #     res = pickle.load(f)
    #     agent1s += [res['p1']]
    #     agent2s += [res['p2']]
    # target_hint, hint_play = marginal_probs(agent1s[0], agent2s[0], episodes=10000)
    # print(target_hint.div(target_hint.sum(axis=1), axis=0))
    # print('/n')
    # print(hint_play.div(hint_play.sum(axis=1), axis=0))
    #
    # target_hint.div(target_hint.sum(axis=1), axis=0).to_csv(f'mp_targethint_{file_des}_{episodes}.csv')
    # hint_play.div(hint_play.sum(axis=1), axis=0).to_csv(f'mp_hintplay_{file_des}_{episodes}.csv')

    agent1s = []
    agent2s = []
    hint_plays = []
    target_hints = []
    hint_play_variance = []
    target_hint_variance = []

    file_des = 'Att3-F'
    episodes = 1000
    filelist = sorted(glob.glob(os.path.join("res/Att3_hs_5_l1_3_l2_3_FalseFalse4000000", "*.pkl")))
    filelist = [filelist[a] for a in  [1,2,3,5,7,8,9,11,14,15,16,]]

    for filename in filelist:
        with open(filename, "rb") as f:
            res = pickle.load(f)
            agent1s += [res['p1']]
            agent2s += [res['p2']]

    for i in range(len(agent1s)):
        target_hint, hint_play = marginal_probs(agent1s[i], agent2s[i], episodes=episodes)
        print(target_hint.div(target_hint.sum(axis=1), axis=0))
        print(hint_play.div(hint_play.sum(axis=1), axis=0))
        target_hints += [target_hint]
        hint_plays += [hint_play]
        target_hint_variance += [target_hint.div(target_hint.sum(axis=1), axis=0).stack().std()]
        hint_play_variance += [hint_play.div(hint_play.sum(axis=1), axis=0).stack().std()]

    target_hint = sum(target_hints)
    hint_play = sum(hint_plays)
    target_hint = target_hint.div(target_hint.sum(axis=1), axis=0)
    hint_play = hint_play.div(hint_play.sum(axis=1), axis=0)
    # target_hint = sum(target_hints) / len(target_hints)
    # hint_play = sum(hint_plays) / len(hint_plays)

    target_hint.to_csv(f'mp_targethint_{file_des}_{len(agent1s)}_{episodes}.csv')
    hint_play.to_csv(f'mp_hintplay_{file_des}_{len(agent1s)}_{episodes}.csv')
    print(np.mean(np.array(target_hint_variance)), np.std(np.array(target_hint_variance)))
    print(np.mean(np.array(hint_play_variance)), np.std(np.array(hint_play_variance)))
