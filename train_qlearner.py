import datetime
import pickle
from collections import namedtuple

import numpy as np
import torch

from hint_play_game import TwoRoundHintGame
from hyperparams import Hp
from qlearner import QLearner

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

hp_train = Hp(hand_size=5,
              nlab1=2,
              nlab2=2,
              shuffle_cards=False,
              opt='adam',
              nepsidoes=1000000,
              batch_size=512,
              eps_scheme={'eps_start': 0.95, 'eps_end': 0.05, 'eps_decay': 100000},
              replay_capacity=100000,
              update_frequency=100,
              )


def obs_to_agent(obs, hp=hp_train):
    o1 = obs[:(1 + hp.hand_size) * (hp.nlab1 + hp.nlab2)]
    o2 = obs[(1 + hp.hand_size) * (hp.nlab1 + hp.nlab2):]
    hand1 = o1[:-(hp.nlab1 + hp.nlab2)]
    hand2 = o2[:-(hp.nlab1 + hp.nlab2)]
    obs1 = np.concatenate([o1, hand2]).flatten()
    obs2 = np.concatenate([o2, hand1]).flatten()
    return obs1, obs2


def train_ff_agents(hp=hp_train, verbose=True):
    num_episodes = hp.nepisodes
    env = TwoRoundHintGame(hp=hp)
    p1 = QLearner(1, env, policy_type='ff', hp=hp)
    p2 = QLearner(1, env, policy_type='ff', hp=hp)

    rewards = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp)
        # P1 select and perform a hint
        # obs1_a1 = torch.tensor(obs1_a1, device=device)
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device))
        obs2, _, _, _ = env.step(a1)
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp)
        # obs2_a2 = torch.tensor(obs2_a2, device=device)
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

        # Update the target network, copying all weights and biases in DQN
        # if i_episode % TARGET_UPDATE == 0:
        #     p1.target_net.load_state_dict(p1.policy_net.state_dict())
        #     p2.target_net.load_state_dict(p2.policy_net.state_dict())

        rewards.append(r.numpy()[0])
        print_num = num_episodes // 50
        if verbose:
            if i_episode > print_num and i_episode % print_num == 0:
                print(datetime.datetime.now(), i_episode, np.array(rewards[-print_num:]).mean())
    print('Training complete')
    p1.memory = None
    p2.memory = None
    result = {'p1': p1, 'p2': p2}
    return result


def train_lat_agents(hp=hp_train, verbose=True, lat_lambda=0.5):
    num_episodes = hp.nepisodes
    env = TwoRoundHintGame(hp=hp)
    p1 = QLearner(1, env, policy_type='ff', hp=hp)
    p2 = QLearner(1, env, policy_type='ff', hp=hp)

    rewards = []
    ips = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp)
        # P1 select and perform a hint
        # obs1_a1 = torch.tensor(obs1_a1, device=device)
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device))
        obs2, _, _, _ = env.step(a1)
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp)
        # obs2_a2 = torch.tensor(obs2_a2, device=device)
        a2 = p2.select_action(torch.tensor([obs2_a2], device=device))
        _, r, _, _ = env.step(a2)
        # Store the transition in memory
        obs1_a1 = torch.tensor([obs1_a1], device=device)
        obs2_a2 = torch.tensor([obs2_a2], device=device)
        a1 = torch.tensor([a1], device=device)
        a2 = torch.tensor([a2], device=device)
        # LAT step
        playable_token = obs1_a1[-(hp.nlab1 + hp.nlab2):].flatten()
        hint_card_token = obs2_a2[-(hp.nlab1 + hp.nlab2):].flatten()
        lat_inner_prod = torch.dot(hint_card_token, playable_token)
        r = torch.tensor([r[0]], device=device)
        p1.memory.push(obs1_a1, a1, None, r + lat_lambda * lat_inner_prod)
        p2.memory.push(obs2_a2, a2, None, r + lat_lambda * lat_inner_prod)

        # Perform one step of the optimization (on the policy network)
        if i_episode % hp.update_frequency == 0:
            p1.optimize_model()
            p2.optimize_model()

        # Update the target network, copying all weights and biases in DQN
        # if i_episode % TARGET_UPDATE == 0:
        #     p1.target_net.load_state_dict(p1.policy_net.state_dict())
        #     p2.target_net.load_state_dict(p2.policy_net.state_dict())

        rewards.append(r.numpy()[0])
        ips.append(lat_inner_prod.numpy()[0])
        print_num = num_episodes // 50
        if verbose:
            if i_episode > print_num and i_episode % print_num == 0:
                print(datetime.datetime.now(), i_episode, np.array(rewards[-print_num:]).mean(),
                      np.array(ips[-print_num:]).mean())
    print('Training complete')
    p1.memory = None
    p2.memory = None
    result = {'p1': p1, 'p2': p2}
    return result


def train_att_agents(hp=hp_train, verbose=True):
    num_episodes = hp.nepisodes
    env = TwoRoundHintGame(hp=hp)
    p1 = QLearner(1, env, policy_type='attention', hp=hp)
    p2 = QLearner(1, env, policy_type='attention', hp=hp)

    rewards = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp)
        obs1_a1 = obs1_a1.reshape(-1, hp.nlab1 + hp.nlab2).T
        obs1_a2 = obs1_a2.reshape(-1, hp.nlab1 + hp.nlab2).T
        # P1 select and perform a hint
        # obs1_a1 = torch.tensor(obs1_a1, device=device)
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device))
        obs2, _, _, _ = env.step(a1)
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp)
        obs2_a1 = obs2_a1.reshape(-1, hp.nlab1 + hp.nlab2).T
        obs2_a2 = obs2_a2.reshape(-1, hp.nlab1 + hp.nlab2).T
        # obs2_a2 = torch.tensor(obs2_a2, device=device)
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

        # Update the target network, copying all weights and biases in DQN
        # if i_episode % TARGET_UPDATE == 0:
        #     p1.target_net.load_state_dict(p1.policy_net.state_dict())
        #     p2.target_net.load_state_dict(p2.policy_net.state_dict())

        rewards.append(r.numpy()[0])
        print_num = num_episodes // 50
        if verbose:
            if i_episode > print_num and i_episode % print_num == 0:
                print(datetime.datetime.now(), i_episode, np.array(rewards[-print_num:]).mean())
    print('Training complete')
    p1.memory = None
    p2.memory = None
    result = {'p1': p1, 'p2': p2}
    return result


def train_att2_agents(hp=hp_train, verbose=True):
    num_episodes = hp.nepisodes
    env = TwoRoundHintGame(hp=hp)
    p1 = QLearner(1, env, policy_type='attention', hp=hp)
    p2 = QLearner(1, env, policy_type='attention', hp=hp)

    rewards = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp)
        obs1_a1 = obs1_a1.reshape(-1, hp.nlab1 + hp.nlab2).T
        obs1_a2 = obs1_a2.reshape(-1, hp.nlab1 + hp.nlab2).T
        # P1 select and perform a hint
        # obs1_a1 = torch.tensor(obs1_a1, device=device)
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device))
        obs2, _, _, _ = env.step(a1)
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp)
        obs2_a1 = obs2_a1.reshape(-1, hp.nlab1 + hp.nlab2).T
        obs2_a2 = obs2_a2.reshape(-1, hp.nlab1 + hp.nlab2).T
        # obs2_a2 = torch.tensor(obs2_a2, device=device)
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

        # Update the target network, copying all weights and biases in DQN
        # if i_episode % TARGET_UPDATE == 0:
        #     p1.target_net.load_state_dict(p1.policy_net.state_dict())
        #     p2.target_net.load_state_dict(p2.policy_net.state_dict())

        rewards.append(r.numpy()[0])
        print_num = num_episodes // 50
        if verbose:
            if i_episode > print_num and i_episode % print_num == 0:
                print(datetime.datetime.now(), i_episode, np.array(rewards[-print_num:]).mean())
    print('Training complete')
    p1.memory = None
    p2.memory = None
    result = {'p1': p1, 'p2': p2}
    return result


if __name__ == '__main__':
    res = train_lat_agents()
    with open(f'res/lat_{str(hp_train)}/' + str(hp_train) + '_' + str(datetime.datetime.now()) + ".pkl",
              'wb') as handle:
        pickle.dump(res, handle)
