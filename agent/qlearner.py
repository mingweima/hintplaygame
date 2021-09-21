import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn

from agent.attention import AttentionModel, AttentionModel2, AttentionModel3
from game.hyperparams import hp_default

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class FeedForwardDNN(nn.Module):

    def __init__(self, input_size, output_size, n_hid=128):
        super(FeedForwardDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, output_size)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = x.to(device)
        return self.model(x.float())


class LSTM(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers=1, n_hid=128):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_size, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, output_size)
        )

    def forward(self, x):
        x = torch.transpose(x, 2, 1).float()
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()  # internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        return self.model(hn)


def get_mechanical(obs, num_cards, card_dim):
    if len(obs.shape) == 2:
        obs.unsqueeze(0)
    hands = int((num_cards - 1) / 2)
    action_space = obs[:, :, hands:-1]
    unique_card = obs[:, :, -1].unsqueeze(2)
    ret = torch.matmul(action_space.transpose(2, 1), unique_card).squeeze(2)
    return ret


class QLearner:
    def __init__(self, player, env, policy_type='ff', hp=hp_default):
        self.player = player
        self.action_space_size = env.action_space.n
        self.obs_space_size = (1 + 2 * hp.hand_size) * (hp.nlab1 + hp.nlab2)
        self.steps_done = 0
        self.memory = ReplayMemory(hp.replay_capacity)
        self.hp = hp
        self.policy_type = policy_type

        if policy_type == 'FF':
            self.policy_net = FeedForwardDNN(self.obs_space_size, self.action_space_size)
            self.policy_net.to(device)
        elif policy_type == 'LSTM':
            card_dim = hp.nlab1 + hp.nlab2
            self.policy_net = LSTM(card_dim, self.action_space_size, self.obs_space_size)
            self.policy_net.to(device)
        elif policy_type == 'Att1':
            num_cards = 1 + 2 * self.hp.hand_size
            card_dim = hp.nlab1 + hp.nlab2
            self.policy_net = AttentionModel(num_cards, card_dim, self.action_space_size)
            self.policy_net.to(device)
        elif policy_type == 'Att2':
            num_cards = 1 + 2 * self.hp.hand_size
            card_dim = hp.nlab1 + hp.nlab2
            self.policy_net = AttentionModel2(num_cards, card_dim, self.action_space_size)
            self.policy_net.to(device)
        elif policy_type == 'Att3':
            num_cards = 1 + 2 * self.hp.hand_size
            card_dim = hp.nlab1 + hp.nlab2
            self.policy_net = AttentionModel3(num_cards, card_dim)
            self.policy_net.to(device)
        elif policy_type == 'mechanical':
            num_cards = 1 + 2 * self.hp.hand_size
            card_dim = hp.nlab1 + hp.nlab2
            self.policy_net = lambda x: get_mechanical(x, num_cards, card_dim)
        else:
            raise ValueError('Policy type unknown!')

        if policy_type != 'mechanical':
            if hp.opt == 'adam':
                self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=hp.lr_adam)
            else:
                self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=hp.lr_sgd)

    def select_action(self, obs, evaluate=False):
        sample = random.random()
        eps_threshold = self.hp.eps_scheme['eps_end'] + (
                self.hp.eps_scheme['eps_start'] - self.hp.eps_scheme['eps_end']) * \
                        math.exp(-1. * self.steps_done / self.hp.eps_scheme['eps_decay'])
        self.steps_done += 1
        if evaluate:
            return self.policy_net(obs).max(1)[1].view(1, 1)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # a = self.policy_net(obs)
                # return self.policy_net(obs).unsqueeze(0).max(1)[1].view(1, 1)
                return self.policy_net(obs).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space_size)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.hp.batch_size:
            return
        transitions = self.memory.sample(self.hp.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        expected_state_action_values = reward_batch

        # Compute Huber loss
        # criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if self.policy_type == 'Att1' or self.policy_type == 'Att2' or self.policy_type == 'Att3':
            self.optimizer.step()
            return
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
