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
        self.actions = torch.zeros((size, concurrent_episodes, 1), device=self.device, dtype=int)
        self.policy_probs = torch.zeros((size, concurrent_episodes, 3), device=self.device)
        self.states_values = torch.zeros((size, concurrent_episodes, 1), device=self.device)
        self.next_possible_states_values = torch.zeros((size, concurrent_episodes, 3), device=self.device)

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
        self.actions = torch.zeros((size, concurrent_episodes, 1), device=self.device, dtype=int)
        self.policy_probs = torch.zeros((size, concurrent_episodes, 3), device=self.device)
        self.states_values = torch.zeros((size, concurrent_episodes, 1), device=self.device)
        self.next_possible_states_values = torch.zeros((size, concurrent_episodes, 3), device=self.device)

    def push(self, state, next_possible_states, reward, actions, policy_probs, states_values, next_possible_states_values):
        self.states[self.pointer] = state.detach()
        self.next_possible_states[self.pointer] = next_possible_states.detach()
        self.rewards[self.pointer] = reward.detach()
        self.actions[self.pointer] = actions.detach()
        self.policy_probs[self.pointer] = policy_probs.detach()
        self.states_values[self.pointer] = states_values.detach()
        self.next_possible_states_values[self.pointer] = next_possible_states_values.detach()

        self.pointer = (self.pointer+1)%self.size
        # print(self.pointer)
        self.iterations += 1

        if self.iterations >= self.size:
            self.filled = True

    def sample(self):
        return

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = actor

        # critic
        self.critic = critic

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        policy_probs = self.actor(state)
        # print(action_probs)
        try:
          dist = Categorical(policy_probs)
        except Exception as e:
          print(policy_probs)
          raise


        # return torch.argmax(policy_probs, dim=-1).detach(), policy_probs.detach()

        action = dist.sample()
        # # action_logprob = dist.log_prob(action)
        #
        return action.detach(), policy_probs.detach()

    def evaluate(self, states):
        policy_probs = self.actor(states)
        dist = Categorical(policy_probs)
        # action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        states_value = self.critic(states)
        # next_possible_states_values = self.critic(next_possible_states.view(-1, *states_shape)).view(-1, 3)

        return dist.probs, states_value, dist_entropy

    def evaluate_next_states(self, states):
        B, A, L, D = states.shape
        states = states.view(-1, L, D)

        # print(self.critic(states).shape)

        next_possible_states_values = self.critic(states).view(B, A)

        return next_possible_states_values


class PPO:
    def __init__(self, actor_critic, lr_actor, lr_critic, gamma, soft_update_ratio, soft_update_period, K_epochs, eps_clip, buffer, mini_batchs_size=16, verbose=False, cuda=False):

        self.verbose = verbose

        self.device = torch.device('cuda:0') if cuda else torch.device('cpu')

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.soft_update_ratio = soft_update_ratio
        self.soft_update_period = soft_update_period

        self.mini_batchs_size = mini_batchs_size

        self.buffer = buffer

        self.policy = actor_critic
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = actor_critic
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.epoch_steps = 1

        self.steps = 0
        self.t_steps = 0

    def calculate_ardae(self, rewards, states_values, next_possible_states_values):
        return (rewards+self.gamma*next_possible_states_values-states_values).detach()

    def select_action(self, state):
        self.steps += 1
        self.t_steps += 1
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(self.device)
            action, policy_probs = self.policy_old.act(state)

        return action, policy_probs

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

        policy_probs_reshaped = self.buffer.policy_probs.view(buffer_length*concurrent_episodes, 3)
        states_values_reshaped = self.buffer.states_values.view(buffer_length*concurrent_episodes, 1)
        next_possible_states_values_reshaped = self.buffer.next_possible_states_values.view(buffer_length*concurrent_episodes, 3)

        batch_indices = torch.randperm(buffer_length*concurrent_episodes, device=self.device).view(-1, mini_batchs_size)

        for steps in range(self.epoch_steps):
            batch_index = batch_indices[steps]

            old_states = torch.index_select(states_reshaped, 0, batch_index).detach()
            old_next_possible_states = torch.index_select(next_possible_states_reshaped, 0, batch_index).detach()
            rewards = torch.index_select(rewards_reshaped, 0, batch_index).detach()

            old_policy_probs = torch.index_select(policy_probs_reshaped, 0, batch_index).detach()
            old_states_values = torch.index_select(states_values_reshaped, 0, batch_index).detach()
            old_next_possible_states_values = torch.index_select(next_possible_states_values_reshaped, 0, batch_index).detach()
            # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            policy_probs, states_values, dist_entropy = self.policy.evaluate(old_states)

            # Compute Advantage
            advantages = self.calculate_ardae(rewards, old_states_values, old_next_possible_states_values)
            advantages_norm = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            # indices = torch.argmax(old_policy_probs, dim=-1, keepdim=True)
            # expected_states_values = torch.gather(advantages, -1, indices) + old_states_values
            expected_states_values = torch.einsum('ba, ba -> b', old_policy_probs, advantages).unsqueeze(1) + old_states_values
            expected_states_values = expected_states_values.detach()


            # Policy related
            ratios = policy_probs/old_policy_probs

            # weighted_advantages = torch.einsum('ba, ba -> b', old_policy_probs, advantages_norm).unsqueeze(1)
            # weighted_advantages = advantages_norm
            # surr1 = torch.einsum('ba, ba -> b', ratios, weighted_advantages)
            # surr2 = torch.einsum('ba, ba -> b', torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip), weighted_advantages)

            indices = torch.argmax(old_policy_probs, dim=-1, keepdim=True)
            action_advantages_norm = torch.gather(advantages_norm, -1, indices)
            action_ratios = torch.gather(ratios, -1, indices)
            surr1 = action_ratios*action_advantages_norm
            surr2 =  torch.clamp(action_ratios, 1-self.eps_clip, 1+self.eps_clip)*action_advantages_norm

            # Loss
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(states_values, expected_states_values) - 0.01*dist_entropy
            loss = loss.mean()
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.policy_old.load_state_dict(self.policy.state_dict())

        if self.steps%self.soft_update_period == 0:
            # Soft update
            policy_state_dict = self.policy.state_dict()
            policy_old_state_dict = self.policy_old.state_dict()

            for i in policy_state_dict.keys():
                policy_old_state = policy_old_state_dict[i]
                policy_state = policy_state_dict[i]
                policy_old_state = (1-self.soft_update_ratio)*policy_old_state + self.soft_update_ratio*policy_state

    def clear(self):
        self.steps = 0
        self.buffer.clear()

    def get_actor_critic(self):
        return self.policy
