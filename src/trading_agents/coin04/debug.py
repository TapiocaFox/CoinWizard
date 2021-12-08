from utils import ConfigLoader
from data_loader import DataLoader
from coin_gym import Gym
from reward import RewardCalculator
from utils import InputDataPreProcessor, calculate_gamma, plot_list, plot_tensor
from ppo import PPO, ActorCritic, ReplayBuffer
from net import CoinNet
from math import floor
from datetime import datetime

import random, json, torch, os, jsbeautifier
import matplotlib.pyplot as plt

config = ConfigLoader('config.json')

trading_agent_name = config.get('trading_agent_name')

trading_agents_files_path = '../../trading_agents_files/'+trading_agent_name

if not os.path.exists(trading_agents_files_path):
    os.makedirs(trading_agents_files_path)

selected_instrument = config.get('selected_instrument')
primary_intrument = config.get('primary_intrument')
granularity_list = config.get('granularity_list')
primary_granularity = config.get('primary_granularity')
input_period_list = config.get('input_period_list')
net_input_period_list = config.get('net_input_period_list')
recent_log_returns_length = config.get('recent_log_returns_length')
from_datetime = config.getDate('train_start_year', 'train_start_month', 'train_start_day', 'start_hour', 'start_min')
to_datetime = config.getDate('train_end_year', 'train_end_month', 'train_end_day', 'end_hour', 'end_min')
test_from_datetime = config.getDate('test_start_year', 'test_start_month', 'test_start_day', 'start_hour', 'start_min')
test_to_datetime = config.getDate('test_end_year', 'test_end_month', 'test_end_day', 'end_hour', 'end_min')
episode_steps = config.get('episode_steps')
epoch_size = config.get('epoch_size')
horizon = config.get('horizon')
t_len = config.get('t_len')
learning_rate = config.get('learning_rate')
k_epochs = config.get('k_epochs')
eps_clip = config.get('eps_clip')
mini_batchs_size = config.get('mini_batchs_size')
concurrent_episodes = config.get('concurrent_episodes')
episodes_rounds = config.get('episodes_rounds')
cuda = config.get('cuda')
device = torch.device('cuda:0') if cuda else torch.device('cpu')

actor_critic = ActorCritic(
    CoinNet(),
    CoinNet(critic=True),
)

if cuda:
    actor_critic.cuda()

ppo_agent = PPO(
    actor_critic,
    learning_rate,
    learning_rate,
    calculate_gamma(horizon, t_len),
    k_epochs,
    eps_clip,
    ReplayBuffer(concurrent_episodes, (96, 14), cuda=cuda),
    mini_batchs_size,
    cuda=cuda
)

gym = Gym(  selected_instrument,
            granularity_list,
            primary_granularity,
            input_period_list,
            from_datetime,
            to_datetime,
            episode_steps,
            RewardCalculator(cuda=cuda),
            concurrent_episodes=concurrent_episodes,
            cuda=cuda
            )


data_pre_processor = InputDataPreProcessor(
    concurrent_episodes,
    net_input_period_list[primary_granularity],
    recent_log_returns_length,
    cuda=cuda
)

(candles, positions) = gym.reset()
print('see', candles[0][concurrent_episodes-1])
data_pre_processor.pushPosition(positions)
print('process', data_pre_processor.process(candles[0]))

(candles, positions), _, _ = gym.step(torch.tensor([2 for i in range(concurrent_episodes)], device=device))
print('see', candles[0][concurrent_episodes-1])
data_pre_processor.pushPosition(positions)
print('process', data_pre_processor.process(candles[0]))
plot_tensor(data_pre_processor.process(candles[0])[0])
print(ppo_agent.select_action(data_pre_processor.process(candles[0])))
(candles, positions), _, _ = gym.step(torch.tensor([2 for i in range(concurrent_episodes)], device=device))
print('see', candles[0][concurrent_episodes-1])
data_pre_processor.pushPosition(positions)
print('process', data_pre_processor.process(candles[0]))
print(ppo_agent.select_action(data_pre_processor.process(candles[0])))
(candles, positions), _, _ = gym.step(torch.tensor([1 for i in range(concurrent_episodes)], device=device))
print('see', candles[0][concurrent_episodes-1])
data_pre_processor.pushPosition(positions)
print('all positions', data_pre_processor.patchAllPositions(candles[0]))
print('process', data_pre_processor.process(candles[0]))
print('get_summary', gym.reward_calculator.get_summary())
