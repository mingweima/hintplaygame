import datetime
import pickle
from collections import namedtuple

import numpy as np
import torch

from hint_play_game import TwoRoundHintGame, Hp
from qlearner import QLearner

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
TARGET_UPDATE = 10

hp = Hp(hand_size=5,
        nlab1=5,
        nlab2=5,
        )


def obs_to_agent(obs, hp=hp):
    o1 = obs[:(1 + hp.hand_size) * (hp.nlab1 + hp.nlab2)]
    o2 = obs[(1 + hp.hand_size) * (hp.nlab1 + hp.nlab2):]
    hand1 = o1[:-(hp.nlab1 + hp.nlab2)]
    hand2 = o2[:-(hp.nlab1 + hp.nlab2)]
    obs1 = np.concatenate([o1, hand2]).flatten()
    obs2 = np.concatenate([o2, hand1]).flatten()
    return obs1, obs2


def train_ff_agents(hp=hp):
    num_episodes = 1000000
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
        if i_episode % 100 == 0:
            p1.optimize_model()
            p2.optimize_model()

        # Update the target network, copying all weights and biases in DQN
        # if i_episode % TARGET_UPDATE == 0:
        #     p1.target_net.load_state_dict(p1.policy_net.state_dict())
        #     p2.target_net.load_state_dict(p2.policy_net.state_dict())

        rewards.append(r.numpy()[0])
        print_num = 10000
        if i_episode > print_num and i_episode % print_num == 0:
            print(datetime.datetime.now(), i_episode, np.array(rewards[-print_num:]).mean())
    print('Training complete')
    result = {'p1': p1, 'p2': p2, 'r': rewards}
    return result


if __name__ == '__main__':
    res = train_ff_agents()
    with open(str(hp) + '_' + str(datetime.datetime.now()), 'wb') as handle:
        pickle.dump(res, handle)
