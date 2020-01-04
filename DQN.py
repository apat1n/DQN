import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Config import config
from Logger import writer
from typing import Tuple, Union


class ER:
    def __init__(self, max_size: int, batch_size: int):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = np.array([(None, None, None, None, None) for _ in range(max_size)])
        self.cur_ind = 0
        self.cur_size = 0

    def append(self, e: Tuple[np.ndarray, Union[np.ndarray, int], float, np.ndarray, bool]):
        """(s, a, r, s', done)"""
        self.buffer[self.cur_ind] = e
        self.cur_ind = (self.cur_ind + 1) % self.max_size
        self.cur_size = max(self.cur_size, self.cur_ind)

    def sample(self):
        indices = np.random.randint(0, self.cur_size, size=self.batch_size)
        return self.buffer[indices]


class Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, out_dim)

    def forward(self, x: torch.FloatTensor):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DQN_discrete:
    def __init__(self,
                 n_state: int,
                 n_action: int,
                 eps: float = 0.05,
                 gamma: float = 0.999,
                 er_batch_size: int = 32,
                 er_max_size: int = 10000,
                 play_before_learn: int = 1000):
        assert er_max_size > play_before_learn, "Need larger ER buffer max_size"
        self.net = Net(n_state, n_action).to(config['device'])
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.replay_buffer = ER(er_max_size, er_batch_size)
        self.play_before_learn = play_before_learn
        self.action_space = range(n_action)
        self.gamma = gamma
        self.eps = eps

    def act(self, state: np.ndarray):
        if self.replay_buffer.cur_size > self.play_before_learn:
            if np.random.rand() <= self.eps:
                action = np.random.choice(self.action_space)
            else:
                state = torch.from_numpy(state).float().to(config['device'])
                with torch.no_grad():
                    values = self.net(state).cpu().numpy()
                action = np.argmax(values)
        else:
            # if experience replay buffer isn't fill make random action
            action = np.random.choice(self.action_space)
        return action

    def append(self, e: Tuple[np.ndarray, Union[np.ndarray, int], float, np.ndarray, bool]):
        self.replay_buffer.append(e)

    def train(self):
        if self.replay_buffer.cur_size > self.play_before_learn:
            y = []
            train_batch = self.replay_buffer.sample()
            s = np.stack(train_batch[:, 0])
            s = torch.from_numpy(s).float().to(config['device'])
            a = np.stack(train_batch[:, 1])
            a = torch.LongTensor(a).to(config['device'])
            s_next = np.stack(train_batch[:, 3])
            s_next = torch.from_numpy(s_next).float().to(config['device'])
            with torch.no_grad():
                s_next_pred = self.net(s_next).cpu().numpy()
            for ind, (_, _, r, _, done) in enumerate(train_batch):
                y.append(r if done else r + self.gamma * np.max(s_next_pred[ind]))

            y = torch.from_numpy(np.array(y)).float()
            pred = self.net(s).gather(1, a.unsqueeze(1))

            self.optimizer.zero_grad()
            loss = torch.sum((y - pred) ** 2)
            loss.backward()
            writer.add_scalar('Loss', loss.detach().cpu().numpy())
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
