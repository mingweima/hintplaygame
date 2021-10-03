import os
import random
import sys
from abc import ABC
from copy import deepcopy

sys.path.append(os.getcwd())
sys.path.append("..")
sys.path.append('path')


import numpy as np
import torch

from game.hyperparams import hp_default

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


def card_token_to_symbol(card_token, hp=hp_default):
    lab1_ind = np.where(card_token.flatten() == 1)[0][0]
    lab2_ind = np.where(card_token.flatten() == 1)[0][1]
    lab1 = hp.label1_list[lab1_ind]
    lab2 = hp.label2_list[lab2_ind - hp.nlab1]
    return lab1 + lab2


def hand_to_symbol(hand, hp=hp_default, highlight=None):
    symbols = []
    for card_ind in range(hp.hand_size):
        card = hand[card_ind * (hp.nlab1 + hp.nlab2):card_ind * (hp.nlab1 + hp.nlab2) + hp.nlab1 + hp.nlab2]
        if highlight is not None and (card == highlight).all():
            symbols.append("\u0332".join(card_token_to_symbol(card, hp=hp)))
        else:
            symbols.append(card_token_to_symbol(
                card,
                hp=hp))
    return symbols


def select_card(hand, card_ind, hp=hp_default):
    """
    Return specific card token from a hand vector (flattened token of cards)
    :param hand: np.array 1D
    :param card_ind: int
    :return: np.array 1D
    """
    return hand[card_ind * (hp.nlab1 + hp.nlab2):card_ind * (hp.nlab1 + hp.nlab2) + hp.nlab1 + hp.nlab2]


def get_rand_card(hp=hp_default):
    """
    Generate random card vector
    :param seed: int
    :return: np.array 1D
    """
    label1 = random.sample(range(hp.nlab1), 1)[0]
    label2 = random.sample(range(hp.nlab2), 1)[0] + hp.nlab1
    card = np.zeros((hp.nlab1 + hp.nlab2))
    card[label1] = 1
    card[label2] = 1
    return card


def get_initial_state(hp=hp_default):
    """
    Generate initial obs assignment
    P1 gets all hands as well as the playable card for P2
    P2 gets all hands as well as an empty card
    """
    info = {}
    o1, o2 = [], []
    for card_ind in range(hp.hand_size):
        card = get_rand_card(hp=hp)
        o1.append(card)
        if hp.same_hand:
            o2.append(card)
        else:
            o2.append(get_rand_card(hp=hp))
    if hp.shuffle_cards:
        random.shuffle(o2)
    playable_card_num = random.sample(range(hp.hand_size), 1)[0]
    o1.append(o2[playable_card_num])
    o2.append(np.zeros((hp.nlab1 + hp.nlab2)))
    o1 = np.array(o1).flatten()
    o2 = np.array(o2).flatten()
    info['playbale_card_num'] = playable_card_num
    return (o1, o2), info


def apply_hint(action, o1, o2, hp=hp_default):
    """
    Take round 1 hint from p1 and inital obs for p1 p2, return new obs (with hints) for p2
    :param action: int, card index
    :param o1: np.array 1D
    :param o2: np.array 1D
    :return o2_new: np.array 1D
    """
    action = select_card(o1[:-(hp.nlab1 + hp.nlab2)], action, hp=hp)  # convert card index to token
    o2_new = deepcopy(o2)
    o2_new[-(hp.nlab1 + hp.nlab2):] = action
    return o2_new


def apply_play_card(action, o1, o2, hp=hp_default):
    """
    Take round 2 action from p2 and obs for p1 and p2, return reward if card is playable.
    :param hp:
    :param action: int, card index
    :param o1: np.array 1D
    :param o2: np.array 1D
    :return r: float
    """
    r = 0.0
    action = select_card(o2[:-(hp.nlab1 + hp.nlab2)], action, hp=hp)  # convert card index to token
    if (action == o1[-(hp.nlab1 + hp.nlab2):]).all():
        r = 1.0
    return r


class TwoRoundHintGame:

    def __init__(self, hp=hp_default):
        # observation is (hand_size) number of cards (own cards) plus 1 special card
        # the special card is the playable card for p1 and the hinted card for p2
        # each card is tokenized to be a 2-hot vector of size (nlab1 + nlab2) where the 1s correspond to that label
        self.hp = hp
        self.o1 = np.zeros(((1 + hp.hand_size) * (hp.nlab1 + hp.nlab2)))
        self.o2 = np.zeros(((1 + hp.hand_size) * (hp.nlab1 + hp.nlab2)))
        self.info = {}
        self.step_count = None
        self.done = False

    def reset(self):
        self.step_count = 0
        self.info = {}
        (self.o1, self.o2), self.info = get_initial_state(hp=self.hp)
        obs = np.array([self.o1.astype(float), self.o2.astype(float)]).flatten()
        self.info['final_reward'] = 0
        self.done = False
        return obs, self.info

    def step(self, action):
        if self.step_count == 0:
            # first step, action is a card hinted by p1
            # second step, action is the card played by p2
            # action is int in range(hand_size) indicating which card of the hand to hint/play
            # reward is zero in this step
            # observation is different for p2, which contains all cards from the players as well as the hinted card
            self.o2 = apply_hint(action, self.o1, self.o2, hp=self.hp)
            r = 0
            self.step_count += 1
            self.info['hint_card_number'] = action
        elif self.step_count == 1:
            r = apply_play_card(action, self.o1, self.o2, hp=self.hp)
            card_played = select_card(self.o2[:-(self.hp.nlab1 + self.hp.nlab2)], action,
                                      hp=self.hp)  # convert card index to token
            self.info['card_played'] = card_played
            self.info['played_card_number'] = action
            self.done = True
        else:
            raise ValueError(f'Step size is {self.step_count} > 1!')
        obs = np.array([self.o1.astype(float), self.o2.astype(float)]).flatten()
        reward = [r, r]
        self.info['final_reward'] += reward[0]
        return obs, reward, self.done, self.info

    def render(self, **kwargs):
        # Underline all cards the same as the playable card
        if self.step_count == 0:
            hand1 = self.o1[:-(self.hp.nlab1 + self.hp.nlab2)]
            hand2 = self.o2[:-(self.hp.nlab1 + self.hp.nlab2)]
            playable_card = self.o1[-(self.hp.nlab1 + self.hp.nlab2):]
            print(
                f'===== Game starts with handsize: {self.hp.hand_size}, nlabl: {self.hp.nlab1}, nlab2: {self.hp.nlab2} =====')
            print(f'Playable hand is: {card_token_to_symbol(playable_card, hp=self.hp)}')
            print(f'Agent 1 hand is: {hand_to_symbol(hand1, hp=self.hp, highlight=playable_card)}')
            print(f'Agent 2 hand is: {hand_to_symbol(hand2, hp=self.hp, highlight=playable_card)}')
        if self.step_count == 1 and not self.done:
            card_hinted = self.o2[-(self.hp.nlab1 + self.hp.nlab2):]
            print(f'Agent 1 hints {card_token_to_symbol(card_hinted, hp=self.hp)} @ ',
                  self.info['hint_card_number'].numpy()[0][0])
        if self.done:
            card_played = self.info['card_played']
            print(f'Agent 2 plays {card_token_to_symbol(card_played, hp=self.hp)} @ ',
                  self.info['played_card_number'].numpy()[0][0])
            print(f'Final reward of game: {self.info["final_reward"]}')
            print('\n')

    def get_card_symbol_in_round(self):
        # Underline all cards the same as the playable card
        if self.step_count == 0:
            playable_card = self.o1[-(self.hp.nlab1 + self.hp.nlab2):]
            return card_token_to_symbol(playable_card, hp=self.hp)
        if self.step_count == 1 and not self.done:
            card_hinted = self.o2[-(self.hp.nlab1 + self.hp.nlab2):]
            return card_token_to_symbol(card_hinted, hp=self.hp)
        if self.done:
            card_played = self.info['card_played']
            return card_token_to_symbol(card_played, hp=self.hp)

