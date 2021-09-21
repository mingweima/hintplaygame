import argparse
import datetime
import glob
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
from train_qlearner import obs_to_agent

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


def teach_ff_agent(teacher_agent,
                   hp_student=hp_train_default,
                   hp_teacher=hp_train_default,
                   verbose=True):
    num_episodes = hp_student.nepisodes
    env = TwoRoundHintGame(hp=hp_student)
    p1 = teacher_agent
    p2 = QLearner(1, env, policy_type='ff', hp=hp_student)

    rewards = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        obs1_a1, _ = obs_to_agent(obs1, hp=hp_teacher)
        # P1 select and perform a hint
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device))
        obs2, _, _, _ = env.step(a1)
        # P2 plays a card
        _, obs2_a2 = obs_to_agent(obs2, hp=hp_student)
        a2 = p2.select_action(torch.tensor([obs2_a2], device=device))
        _, r, _, _ = env.step(a2)
        # Store the transition in memory
        obs2_a2 = torch.tensor([obs2_a2], device=device)
        a2 = torch.tensor([a2], device=device)
        r = torch.tensor([r[0]], device=device)
        p2.memory.push(obs2_a2, a2, None, r)

        # Perform one step of the optimization (on the policy network)
        if i_episode % hp_student.update_frequency == 0:
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


def teach_att3_agents(teacher_agent,
                      hp_student=hp_train_default,
                      hp_teacher=hp_train_default,
                      verbose=True):
    num_episodes = hp_student.nepisodes
    env = TwoRoundHintGame(hp=hp_student)
    p1 = teacher_agent
    p2 = QLearner(1, env, policy_type='attention3', hp=hp_student)

    rewards = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        obs1_a1, _ = obs_to_agent(obs1, hp=hp_teacher)
        obs1_a1 = obs1_a1.reshape(-1, hp_teacher.nlab1 + hp_teacher.nlab2).T
        # P1 select and perform a hint
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device))
        obs2, _, _, _ = env.step(a1)
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp_student)
        obs2_a2 = obs2_a2.reshape(-1, hp_student.nlab1 + hp_student.nlab2).T
        a2 = p2.select_action(torch.tensor([obs2_a2], device=device))
        _, r, _, _ = env.step(a2)
        # Store the transition in memory
        obs2_a2 = torch.tensor([obs2_a2], device=device)
        a2 = torch.tensor([a2], device=device)
        r = torch.tensor([r[0]], device=device)
        p2.memory.push(obs2_a2, a2, None, r)

        # Perform one step of the optimization (on the policy network)
        if i_episode % hp_student.update_frequency == 0:
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


def teach_agents(teacher_agent,
                 hp_student=hp_train_default,
                 hp_teacher=hp_train_default,
                 verbose=True, ):
    num_episodes = hp_student.nepisodes
    env = TwoRoundHintGame(hp=hp_student)
    p1 = teacher_agent
    p2 = QLearner(1, env, policy_type=hp_student.agent_type, hp=hp_student)

    rewards = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        obs1, info = env.reset()
        obs1_a1, obs1_a2 = obs_to_agent(obs1, hp=hp_teacher)
        if hp_teacher.agent_type != 'FF':
            obs1_a1 = obs1_a1.reshape(-1, hp_teacher.nlab1 + hp_teacher.nlab2).T
        # P1 select and perform a hint
        a1 = p1.select_action(torch.tensor([obs1_a1], device=device))
        obs2, _, _, _ = env.step(a1)
        # P2 plays a card
        obs2_a1, obs2_a2 = obs_to_agent(obs2, hp=hp_student)
        if hp_student.agent_type != 'FF':
            obs2_a2 = obs2_a2.reshape(-1, hp_student.nlab1 + hp_student.nlab2).T
        a2 = p2.select_action(torch.tensor([obs2_a2], device=device))
        _, r, _, _ = env.step(a2)
        # Store the transition in memory
        obs1_a1 = torch.tensor([obs1_a1], device=device)
        obs2_a2 = torch.tensor([obs2_a2], device=device)
        a1 = torch.tensor([a1], device=device)
        a2 = torch.tensor([a2], device=device)
        r = torch.tensor([r[0]], device=device)
        p2.memory.push(obs2_a2, a2, None, r)

        # Perform one step of the optimization (on the policy network)
        if i_episode % hp_student.update_frequency == 0:
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
    parser.add_argument('--teacher_type', type=str, default='Att3')

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

    hp_teacher = Hp(hand_size=args.hand_size,
                    nlab1=args.nlab1,
                    nlab2=args.nlab2,
                    shuffle_cards=args.shuffle_cards,
                    agent_type=args.teacher_type,
                    opt='adam',
                    nepisodes=args.nepisodes,
                    batch_size=args.batch_size,
                    eps_scheme={'eps_start': 0.95, 'eps_end': 0.01, 'eps_decay': 50000},
                    replay_capacity=args.replay_capacity,
                    update_frequency=args.update_frequency,
                    )

    hp_train = hp_train_current

    teacher_path = f'res/{hp_teacher}'
    agent1s = []
    for filename in glob.glob(os.path.join(teacher_path, "*.pkl")):
        with open(filename, "rb") as f:
            res = pickle.load(f)
            agent1s += [res['p1']]
    teacher = agent1s[0]

    res = teach_agents(teacher, hp_teacher=hp_teacher,
                       hp_student=hp_train, verbose=True)

    save_dir = f'res/teach/{hp_train}'
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    hp_train.log(res_path=save_dir, file_name='hp_student.txt')
    hp_teacher.log(res_path=save_dir, file_name='hp_teacher.txt')

    with open((os.path.join(save_dir, str(datetime.datetime.now()) + ".pkl")), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
