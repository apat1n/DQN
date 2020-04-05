import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from Config import config
from typing import Tuple, Union, Any


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
        indices = random.sample(range(0, self.cur_size), self.batch_size)
        s, a, r, s_next, done = zip(*self.buffer[indices])
        return np.stack(s), np.stack(a), np.stack(r), np.stack(s_next), np.stack(done)


class Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, out_dim)
        # xavier init
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, x: torch.FloatTensor):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DQN:
    def __init__(self,
                 n_state: int,
                 n_action: int,
                 writer: Any = None,
                 eps_start: float = 0.95,
                 eps_final: float = 0.001,
                 eps_decay: int = 2000,
                 gamma: float = 0.99,
                 er_batch_size: int = 128,
                 er_max_size: int = 10000,
                 target_freq: int = 2000,
                 play_before_learn: int = 128):
        assert er_max_size > play_before_learn >= er_batch_size, \
            "Need larger ER buffer max_size"
        self.net = Net(n_state, n_action).to(config['device'])
        self.target_net = Net(n_state, n_action).to(config['device'])
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.replay_buffer = ER(er_max_size, er_batch_size)
        self.play_before_learn = play_before_learn
        self.action_space = range(n_action)
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.target_freq = target_freq
        self.writer = writer
        self.frame_ind = 0

    def get_eps(self):
        return self.eps_final + (self.eps_start - self.eps_final) * np.exp(-1. * self.frame_ind / self.eps_decay)

    def act(self, state: np.ndarray):
        if self.replay_buffer.cur_size > self.play_before_learn:
            if np.random.rand() <= self.get_eps():
                action = np.random.choice(self.action_space)
            else:
                state = torch.from_numpy(state).float().to(config['device'])
                with torch.no_grad():
                    values = self.net(state).cpu().numpy()
                action = np.argmax(values)
            self.frame_ind += 1
        else:
            # if experience replay buffer isn't fill make random action
            action = np.random.choice(self.action_space)
        return action

    def append(self, e: Tuple[np.ndarray, Union[np.ndarray, int], float, np.ndarray, bool]):
        self.replay_buffer.append(e)

    def train(self):
        if self.replay_buffer.cur_size > self.play_before_learn:
            if self.frame_ind % self.target_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

            s, a, r, s_next, done = self.replay_buffer.sample()
            s = torch.from_numpy(s).float().to(config['device'])
            a = torch.from_numpy(a).long().to(config['device'])
            r = torch.from_numpy(r).float().to(config['device'])
            s_next = torch.from_numpy(s_next).float().to(config['device'])
            done = torch.from_numpy(done).float().to(config['device'])
            with torch.no_grad():
                s_next_pred = self.target_net(s_next).max(1)[0]
            y = r + self.gamma * s_next_pred * (1 - done)
            pred = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = torch.sum((y - pred) ** 2)

            if self.writer is not None:
                self.writer.add_scalar('Loss', loss.detach().cpu().numpy(), self.frame_ind)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
