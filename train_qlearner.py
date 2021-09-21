import argparse
import datetime
import os
import pickle
import sys
from collections import namedtuple

import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append("")

from game.hint_play_game import TwoRoundHintGame
from game.hyperparams import Hp
from agent.qlearner import QLearner

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

hp_train_default = Hp(hand_size=5,
                      nlab1=4,
                      nlab2=4,
                      shuffle_cards=False,
                      opt='adam',
                      nepisodes=500000,
                      batch_size=512,
                      eps_scheme={'eps_start': 0.95, 'eps_end': 0.01, 'eps_decay': 50000},
                      replay_capacity=200000,
                      update_frequency=100,
                      )


def obs_to_agent(obs, hp=hp_train_default):
    o1 = obs[:(1 + hp.hand_size) * (hp.nlab1 + hp.nlab2)]
    o2 = obs[(1 + hp.hand_size) * (hp.nlab1 + hp.nlab2):]
    hand1 = o1[:-(hp.nlab1 + hp.nlab2)]
    hand2 = o2[:-(hp.nlab1 + hp.nlab2)]
    # shuffle hands
    """
    hand1_permute = hand1.reshape(-1, hp.nlab1 + hp.nlab2)
    np.random.shuffle(hand1_permute)
    hand1 = hand1_permute.flatten()
    hand2_permute = hand2.reshape(-1, hp.nlab1 + hp.nlab2)
    np.random.shuffle(hand2_permute)
    hand2 = hand2_permute.flatten()
    """

    obs1 = np.concatenate([hand2, o1]).flatten()
    obs2 = np.concatenate([hand1, o2]).flatten()
    return obs1, obs2


def train_agents(hp=hp_train_default, verbose=True):
    num_episodes = hp.nepisodes
    env = TwoRoundHintGame(hp=hp)
    policy = hp.agent_type
    p1 = QLearner(1, env, policy_type=policy, hp=hp)
    p2 = QLearner(1, env, policy_type=policy, hp=hp)

    rewards = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp)
        if policy != 'FF':
            obs1_a1 = obs1_a1.reshape(-1, hp.nlab1 + hp.nlab2).T
        # P1 select and perform a hint
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device))
        obs2, _, _, _ = env.step(a1)
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp)
        if policy != 'FF':
            obs2_a2 = obs2_a2.reshape(-1, hp.nlab1 + hp.nlab2).T
        a2 = p2.select_action(torch.tensor([obs2_a2], device=device))
        _, r, _, _ = env.step(a2)
        # Store the transition in memory
        obs1_a1 = torch.tensor([obs1_a1], device=device)
        obs2_a2 = torch.tensor([obs2_a2], device=device)
        a1 = torch.tensor([a1], device=device)
        a2 = torch.tensor([a2], device=device)
        r = torch.tensor([r[0]], device=device)
        p1.memory.push(obs1_a1, a1, None, r)
        p2.memory.push(obs2_a2, a2, None, r)

        # Perform one step of the optimization (on the policy network)
        if i_episode % hp.update_frequency == 0:
            p1.optimize_model()
            p2.optimize_model()

        rewards.append(r.cpu().numpy()[0])
        print_num = num_episodes // 100
        if verbose:
            if i_episode > print_num and i_episode % print_num == 0:
                print(datetime.datetime.now(), i_episode, np.array(rewards[-print_num:]).mean())
    print('Training complete')
    p1.memory = None
    p2.memory = None
    result = {'p1': p1, 'p2': p2, 'rewards': rewards}
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hand_size', type=int, default=5)
    parser.add_argument('--nlab1', type=int, default=3)
    parser.add_argument('--nlab2', type=int, default=3)
    parser.add_argument('--shuffle_cards', type=bool, default=False)
    parser.add_argument('--agent_type', type=str, default='Att3')
    parser.add_argument('--nepisodes', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--replay_capacity', type=int, default=200000)
    parser.add_argument('--update_frequency', type=int, default=100)

    args = parser.parse_args()

    # Att: nepsi
    # Att training params
    hp_train_current = Hp(hand_size=args.hand_size,
                          nlab1=args.nlab1,
                          nlab2=args.nlab2,
                          shuffle_cards=args.shuffle_cards,
                          agent_type=args.agent_type,
                          opt='adam',
                          nepisodes=args.nepisodes,
                          batch_size=args.batch_size,
                          eps_scheme={'eps_start': 0.95, 'eps_end': 0.01, 'eps_decay': 50000},
                          replay_capacity=args.replay_capacity,
                          update_frequency=args.update_frequency,
                          )

    # FF training params
    # hp_train_ff = Hp(hand_size=5,
    #                  nlab1=3,
    #                  nlab2=3,
    #                  shuffle_cards=False,
    #                  opt='adam',
    #                  nepisodes=2000000,
    #                  batch_size=512,
    #                  eps_scheme={'eps_start': 0.95, 'eps_end': 0.01, 'eps_decay': 50000},
    #                  replay_capacity=200000,
    #                  update_frequency=100,
    #                  )
        
    if args.agent_type in ['Att3', 'Att2', 'Att1', 'FF', 'LSTM']:
        hp_train = hp_train_current
        res = train_agents(hp=hp_train)
    else:
        raise ValueError("Agent not found in base!")

    save_dir = f'res/{hp_train}'
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    hp_train.log(res_path=save_dir)

    with open((os.path.join(save_dir, str(datetime.datetime.now()) + ".pkl")), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
