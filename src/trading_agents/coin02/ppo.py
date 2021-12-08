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
        self.policy_probs = torch.zeros((size, concurrent_episodes, 3), device=self.device)
        self.rewards = torch.zeros((size, concurrent_episodes, 3), device=self.device)
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
        self.policy_probs = torch.zeros((size, concurrent_episodes, 3), device=self.device)
        self.rewards = torch.zeros((size, concurrent_episodes, 3), device=self.device)
        self.actions = torch.zeros((size, concurrent_episodes, 1), device=self.device)

    def push(self, state, next_possible_states, policy_probs, reward, actions):
        self.states[self.pointer] = state
        self.next_possible_states[self.pointer] = next_possible_states
        self.policy_probs[self.pointer] = policy_probs
        self.rewards[self.pointer] = reward
        self.actions[self.pointer] = actions

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
        dist = Categorical(policy_probs)

        # return torch.argmax(policy_probs, dim=-1).detach(), policy_probs.detach()

        action = dist.sample()
        # action_logprob = dist.log_prob(action)

        return action.detach(), dist.probs.detach()

    def evaluate(self, states):
        policy_probs = self.actor(states)
        dist = Categorical(policy_probs)
        # action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        states_value = self.critic(states)
        # next_possible_states_values = self.critic(next_possible_states.view(-1, *states_shape)).view(-1, 3)

        return dist.probs, states_value, dist_entropy

    def evaluate_old(self, states, next_possible_states):
        states_shape = (next_possible_states.shape[2], next_possible_states.shape[3])
        states_value = self.critic(states)
        next_possible_states_values = self.critic(next_possible_states.view(-1, *states_shape)).view(-1, 3)

        return states_value.detach(), next_possible_states_values.detach()


class PPO:
    def __init__(self, actor_critic, lr_actor, lr_critic, gamma, K_epochs, epoch_size, eps_clip, buffer, mini_batchs_size=16, verbose=False, cuda=False):

        self.verbose = verbose

        self.device = torch.device('cuda:0') if cuda else torch.device('cpu')

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.epoch_size = epoch_size

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

    def calculate_ardae(self, rewards, states_values, next_possible_states_values):
        return (rewards+self.gamma*next_possible_states_values-states_values).detach()


    def select_action(self, state):
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(self.device)
            action, policy_probs = self.policy_old.act(state)

        return action, policy_probs


    def update(self):

        # # Monte Carlo estimate of returns
        # rewards = []
        # discounted_reward = torch.zeros(self.buffer.rewards[0].shape[0], device=self.device)
        # for reward in reversed(self.buffer.rewards):
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)
        # # Normalizing the rewards
        # rewards = torch.stack(rewards)
        # rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 1e-7)
        # rewards_list = rewards.split(1, dim=0)

        # convert list to tensor
        # old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        # old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        # old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        mini_batchs_size = self.mini_batchs_size
        buffer_length = self.buffer.size
        concurrent_episodes = self.buffer.concurrent_episodes
        state_size= self.buffer.state_size
        epoch_size = (buffer_length*concurrent_episodes)//mini_batchs_size

        states_reshaped = self.buffer.states.view(buffer_length*concurrent_episodes, *state_size)
        next_possible_states_reshaped = self.buffer.next_possible_states.view(buffer_length*concurrent_episodes, 3, *state_size)
        policy_probs_reshaped = self.buffer.policy_probs.view(buffer_length*concurrent_episodes, 3)
        rewards_reshaped = self.buffer.rewards.view(buffer_length*concurrent_episodes, 3)

        # Optimize policy for K epochs
        if self.verbose:
            print('PPO training [ ', end='', flush=True)

        for _ in range(self.K_epochs):

            # Random permutation
            batch_indices = torch.randperm(buffer_length*concurrent_episodes, device=self.device).view(-1, mini_batchs_size)

            for steps in range(epoch_size):
                batch_index = batch_indices[steps]
                # old_states = self.buffer.states.view(buffer_length*concurrent_episodes, -1, -1)[batch_index, :, :]
                # old_actions = self.buffer.next_possible_states.view(buffer_length*concurrent_episodes, -1, -1, -1)[batch_index, :, :, :]
                # old_logprobs = self.buffer.policy_probs.view(buffer_length*concurrent_episodes, -1)[batch_index, :]
                # old_logprobs = self.buffer.rewards.view(buffer_length*concurrent_episodes, -1)[batch_index, :]
                old_states = torch.index_select(states_reshaped, 0, batch_index).detach()
                old_next_possible_states = torch.index_select(next_possible_states_reshaped, 0, batch_index).detach()
                old_policy_probs = torch.index_select(policy_probs_reshaped, 0, batch_index).detach()
                rewards = torch.index_select(rewards_reshaped, 0, batch_index).detach()

                # raise
                # Evaluating old actions and values
                states_old_values, next_possible_states_values = self.policy_old.evaluate_old(old_states, old_next_possible_states)

                policy_probs, states_values, dist_entropy = self.policy.evaluate(old_states)


                # print(old_states.shape)
                # print(old_next_possible_states.shape)
                # print(old_policy_probs.shape)
                # print(rewards.shape)
                # print()
                # print(policy_probs.shape)
                # print(states_values.shape)
                # print(next_possible_states_values.shape)
                # print(dist_entropy.shape)



                # Finding the ratio (pi_theta / pi_theta__old)
                # ratios = torch.exp(policy_probs - old_policy_probs.detach())
                ratios = policy_probs/old_policy_probs
                # ratios = policy_probs

                # Finding Surrogate Loss
                advantages = self.calculate_ardae(rewards, states_old_values, next_possible_states_values)
                weighted_advantages = torch.einsum('ba, ba -> b', policy_probs, advantages).unsqueeze(1)
                # weighted_advantages = torch.einsum('ba, ba -> b', old_policy_probs, advantages).unsqueeze(1)
                # # print(advantages.shape)
                print(advantages)
                # # print(ratios.shape)
                # surr1 = torch.einsum('ba, ba -> b', ratios, weighted_advantages)
                # surr2 = torch.einsum('ba, ba -> b', torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip), weighted_advantages)

                # match states_value tensor dimensions with reward tensor
                target_states_values = torch.einsum('ba, ba -> b', old_policy_probs, advantages).unsqueeze(1) + states_old_values
                target_states_values = target_states_values.detach()
                # print(target_states_values)
                states_values = states_values

                # print(surr1.shape)
                # print(surr2.shape)
                # print(old_policy_probs.shape)
                # print(advantages.shape)
                # print(states_values.shape)
                # print(target_states_values.shape)
                # raise

                # final loss of clipped objective PPO
                # loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(states_values, target_states_values) - 0.01*dist_entropy
                loss = -weighted_advantages + 0.5*self.MseLoss(states_values, target_states_values) - 0.01*dist_entropy

                # take gradient step
                loss = loss.mean()

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.verbose:
                print('>', end='', flush=True)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        # self.buffer.clear()

        if self.verbose:
            print(' ]')

    def clear(self):
        self.buffer.clear()

    def get_actor_critic(self):
        return self.policy
