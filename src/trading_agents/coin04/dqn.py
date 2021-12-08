import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
# from collections import namedtuple
#
# Transition = namedtuple('Transition', ('state', 'next_possible_states', 'log_prob', 'rewards'))

class ReplayBuffer:
    def __init__(self, concurrent_episodes, state_size, size=480, cuda=False):
        self.device = torch.device('cuda:0') if cuda else torch.device('cpu')
        self.concurrent_episodes = concurrent_episodes
        self.state_size = state_size
        self.pointer = 0
        self.iterations = 0
        self.size = size
        self.filled = False
        # ('state', 'next_possible_states', 'log_prob', 'rewards')
        self.states = torch.zeros((size, concurrent_episodes, *state_size), device=self.device)
        self.next_possible_states = torch.zeros((size, concurrent_episodes, 3, *state_size), device=self.device)
        self.rewards = torch.zeros((size, concurrent_episodes, 3), device=self.device)
        self.q_values = torch.zeros((size, concurrent_episodes, 3), device=self.device)
        self.actions = torch.zeros((size, concurrent_episodes, 1), device=self.device, dtype=int)

    def clear(self):
        size = self.size
        concurrent_episodes = self.concurrent_episodes
        state_size = self.state_size
        self.pointer = 0
        self.filled = False
        self.iterations = 0
        self.states = torch.zeros((size, concurrent_episodes, *state_size), device=self.device)
        self.next_possible_states = torch.zeros((size, concurrent_episodes, 3, *state_size), device=self.device)
        self.rewards = torch.zeros((size, concurrent_episodes, 3), device=self.device)
        self.q_values = torch.zeros((size, concurrent_episodes, 3), device=self.device)
        self.actions = torch.zeros((size, concurrent_episodes, 1), device=self.device, dtype=int)

    def push(self, state, next_possible_states, q_value, reward, actions):
        self.states[self.pointer] = state
        self.next_possible_states[self.pointer] = next_possible_states
        self.rewards[self.pointer] = reward
        self.q_values[self.pointer] = q_value
        self.actions[self.pointer] = actions

        self.pointer = (self.pointer+1)%self.size
        # print(self.pointer)
        self.iterations += 1

        if self.iterations >= self.size:
            self.filled = True

    def sample(self):
        return

class QNet(nn.Module):
    def __init__(self, model, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        super(QNet, self).__init__()
        self.epsilon = nn.Parameter(torch.tensor(epsilon))
        self.epsilon_min = nn.Parameter(torch.tensor(epsilon_min))
        self.epsilon_decay = nn.Parameter(torch.tensor(epsilon_decay))

        # model
        self.model = model

    def forward(self):
        raise NotImplementedError

    def act(self, states):
        q_values = self.model(states)
        # TODO: epsilon greedy
        actions = torch.argmax(q_values, dim=-1)
        return actions.detach(), q_values

    def evaluate(self, states):
        q_values = self.model(states)
        return q_values

    def evaluate_next_states(self, states):
        B, A, L, D = states.shape
        states = states.view(-1, L, D)
        # print(self.model(states).shape)
        q_values = self.model(states).view(B, A, 3)
        # print(q_values)
        return q_values

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

class DQN:
    def __init__(self, q_net, learning_rate, gamma, K_epochs, eps_clip, buffer, soft_update_ratio, soft_update_period, mini_batchs_size=16, verbose=False, cuda=False):
        # super(DQN, self).__init__()

        self.verbose = verbose

        self.device = torch.device('cuda:0') if cuda else torch.device('cpu')

        self.gamma = gamma
        self.soft_update_ratio = soft_update_ratio
        self.soft_update_period = soft_update_period
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.epoch_steps = 1


        self.mini_batchs_size = mini_batchs_size

        self.buffer = buffer

        self.policy_net = q_net
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy_net.parameters(), 'lr': learning_rate}
                    ])

        self.target_net = q_net
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.MseLoss = nn.MSELoss()

        self.steps = 0

        self.t_steps = 0

    def select_action(self, state):
        self.t_steps += 1
        self.steps += 1
        with torch.no_grad():
            action, q_values = self.policy_net.act(state)

        return action.int(), q_values


    def update(self):

        mini_batchs_size = self.mini_batchs_size
        buffer_length = self.buffer.size
        concurrent_episodes = self.buffer.concurrent_episodes
        state_size= self.buffer.state_size
        epoch_size = (buffer_length*concurrent_episodes)//mini_batchs_size

        states_reshaped = self.buffer.states.view(buffer_length*concurrent_episodes, *state_size)
        next_possible_states_reshaped = self.buffer.next_possible_states.view(buffer_length*concurrent_episodes, 3, *state_size)
        rewards_reshaped = self.buffer.rewards.view(buffer_length*concurrent_episodes, 3)

        # Optimize policy for K epochs
        if self.verbose:
            print('DQN training [ ', end='', flush=True)

        for _ in range(self.K_epochs):

            # Random permutation
            batch_indices = torch.randperm(buffer_length*concurrent_episodes, device=self.device).view(-1, mini_batchs_size)

            for steps in range(epoch_size):
                batch_index = batch_indices[steps]

                old_states = torch.index_select(states_reshaped, 0, batch_index).detach()
                old_next_possible_states = torch.index_select(next_possible_states_reshaped, 0, batch_index).detach()
                rewards = torch.index_select(rewards_reshaped, 0, batch_index).detach()

                # Compute policy Q-value
                policy_q_values = self.policy_net.evaluate(old_states)

                # Compute target Q-value
                # Target net for value evaluating
                target_next_states_q_values = self.target_net.evaluate_next_states(old_next_possible_states)
                target_next_states_q_values = target_next_states_q_values.detach()
                # print(target_next_states_q_values.shape)

                # Policy net for indices selecting
                policy_next_states_q_values = self.policy_net.evaluate_next_states(old_next_possible_states)
                indices = torch.argmax(policy_next_states_q_values, dim=-1, keepdim=True)

                # Calculate the value
                target_next_states_expected_q_values = torch.gather(target_next_states_q_values, -1, indices).squeeze(-1)

                expected_q_values = target_next_states_expected_q_values*self.gamma + rewards

                loss = self.MseLoss(policy_q_values, expected_q_values)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.verbose:
                print('>', end='', flush=True)

        # Soft update

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for i in target_net_state_dict.keys():
            target_net_state = target_net_state_dict[i]
            policy_net_state = policy_net_state_dict[i]
            target_net_state = (1-self.soft_update_ratio)*target_net_state + self.soft_update_ratio*policy_net_state

        # # Copy new weights into old policy
        # self.target_net.load_state_dict(self.policy_net.state_dict())

        # clear buffer
        # self.buffer.clear()

        if self.verbose:
            print(' ]')

    def step(self):
        if not self.buffer.filled:
            return

        mini_batchs_size = self.mini_batchs_size
        buffer_length = self.buffer.size
        concurrent_episodes = self.buffer.concurrent_episodes
        state_size= self.buffer.state_size

        states_reshaped = self.buffer.states.view(buffer_length*concurrent_episodes, *state_size)
        next_possible_states_reshaped = self.buffer.next_possible_states.view(buffer_length*concurrent_episodes, 3, *state_size)
        rewards_reshaped = self.buffer.rewards.view(buffer_length*concurrent_episodes, 3)

        batch_indices = torch.randperm(buffer_length*concurrent_episodes, device=self.device).view(-1, mini_batchs_size)

        for steps in range(self.epoch_steps):
            batch_index = batch_indices[steps]

            old_states = torch.index_select(states_reshaped, 0, batch_index).detach()
            old_next_possible_states = torch.index_select(next_possible_states_reshaped, 0, batch_index).detach()
            rewards = torch.index_select(rewards_reshaped, 0, batch_index).detach()
            # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            rewards = rewards

            # Compute target Q-value
            # Target net for value evaluating
            target_next_states_q_values = self.target_net.evaluate_next_states(old_next_possible_states)
            target_next_states_q_values = target_next_states_q_values.detach()
            # print(target_next_states_q_values.shape)

            # Policy net for indices selecting
            policy_next_states_q_values = self.policy_net.evaluate_next_states(old_next_possible_states)
            indices = torch.argmax(policy_next_states_q_values, dim=-1, keepdim=True)

            # Calculate the value
            target_next_states_expected_q_values = torch.gather(target_next_states_q_values, -1, indices).squeeze(-1)

            expected_q_values = target_next_states_expected_q_values*self.gamma + rewards

            # expected_q_values_mean = expected_q_values.mean().detach()
            # expected_q_values_std = expected_q_values.std().detach()
            #
            # expected_q_values = (expected_q_values - expected_q_values_mean) / (expected_q_values_std + 1e-7)

            # Compute policy Q-value
            policy_q_values = self.policy_net.evaluate(old_states)
            # policy_q_values = (policy_q_values - expected_q_values_mean) / (expected_q_values_std + 1e-7)

            loss = self.MseLoss(policy_q_values, expected_q_values)

            loss.backward()

            # torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), 1)

            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.steps%self.soft_update_period == 0:
            # Soft update
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()

            for i in target_net_state_dict.keys():
                target_net_state = target_net_state_dict[i]
                policy_net_state = policy_net_state_dict[i]
                target_net_state = (1-self.soft_update_ratio)*target_net_state + self.soft_update_ratio*policy_net_state

    def clear(self):
        self.steps = 0
        self.buffer.clear()
