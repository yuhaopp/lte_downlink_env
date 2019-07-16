import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from simulator import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


def plot_reward(name, reward_list, frame_idx):
    plt.figure(figsize=(10, 10))
    plt.plot(reward_list)
    plt.savefig('{}_{}.eps'.format(name, int(frame_idx / 10000)))
    plt.close()


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.num_actions = num_actions

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

    def add_noise(self, action):
        sigma = 0.3
        action_dim = self.num_actions
        low = 0
        high = 1
        state = sigma * np.random.randn(action_dim)
        action = np.clip(action + state, low, high)
        return action

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state).detach().cpu().numpy()[0]
        action = self.add_noise(action)
        return action

    def get_action_without_noise(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state).detach().cpu().numpy()[0]
        return action


class DDPG:
    def __init__(self, ue_arrival_rate=0.03, episode_tti=200.0):
        self.env = Airview(ue_arrival_rate, episode_tti)
        self.ou_noise = OUNoise(self.env.action_space)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.hidden_dim = 64

        self.value_net = ValueNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)

        self.target_value_net = ValueNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        value_lr = 1e-3
        policy_lr = 1e-4

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.MSELoss()

        self.replay_buffer_size = 100000
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.max_frames = int(episode_tti * 1000)
        self.frame_idx = 0
        self.batch_size = 64
        self.average_reward_list = []
        self.transmit_rate_list = []
        self.num_all_users_list = []
        self.num_selected_users_list = []

    def ddpg_update(self,
                    batch_size,
                    gamma=0.99,
                    min_value=-np.inf,
                    max_value=np.inf,
                    soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def run(self):
        while self.frame_idx < self.max_frames:
            state_list = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                self.frame_idx += 1
                action_list = []
                for state in state_list:
                    action_list.append(self.policy_net.get_action(state))
                mcs_list = [np.argmax(action) + 1 for action in action_list]
                updated_state_list, next_state_list, reward_list, done, all_buffer, num_all_users, num_selected_users, one_step_reward = self.env.step(
                    mcs_list)

                for i in range(len(state_list)):
                    self.replay_buffer.push(state_list[i], action_list[i], reward_list[i], updated_state_list[i], done)
                if len(self.replay_buffer) > self.batch_size:
                    self.ddpg_update(self.batch_size)

                state_list = next_state_list
                episode_reward += one_step_reward
                self.average_reward_list.append(episode_reward / self.frame_idx)
                self.transmit_rate_list.append(episode_reward / all_buffer)
                self.num_all_users_list.append(num_all_users)
                self.num_selected_users_list.append(num_selected_users)
                if self.frame_idx % 1000 == 0:
                    print(self.frame_idx)
                    print('current reward: {}'.format(episode_reward / self.frame_idx))

                if self.frame_idx % 200000 == 0:
                    time = str(datetime.datetime.now())
                    plot_reward('ddpg_policy_reward_{}'.format(time), self.average_reward_list, self.frame_idx)
                    log = open("train_ddpg_policy_result_{}_{}.txt".format(self.frame_idx, time), "w")
                    log.write(str(self.average_reward_list))
                    log.write('\n')
                    log.write(str(self.transmit_rate_list))
                    log.write('\n')
                    log.write(str(self.num_all_users_list))
                    log.write('\n')
                    log.write(str(self.num_selected_users_list))
                    log.close()

        torch.save(self.policy_net, 'ddpg_policy_net.pth')
        return self.average_reward_list
