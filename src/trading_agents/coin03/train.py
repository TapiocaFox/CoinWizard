from utils import ConfigLoader
from data_loader import DataLoader
from coin_gym import Gym
from reward import RewardCalculator
from utils import InputDataPreProcessor, calculate_gamma, plot_list, pretty_print, plot_tensor
from dqn import DQN, QNet, ReplayBuffer
from net import CoinNet
from math import floor
from datetime import datetime

import random, torch, os, copy
import matplotlib.pyplot as plt

config = ConfigLoader('config.json')

print('Config loaded.')
pretty_print(config.config)

trading_agent_name = config.get('trading_agent_name')

trading_agents_files_path = '../../trading_agents_files/'+trading_agent_name

if not os.path.exists(trading_agents_files_path):
    os.makedirs(trading_agents_files_path)

selected_instrument = config.get('selected_instrument')
selected_instrument_spread = config.get('selected_instrument_spread')
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
update_n_steps = config.get('update_n_steps')
soft_update_ratio = config.get('soft_update_ratio')
soft_update_period = config.get('soft_update_period')
horizon = config.get('horizon')
t_len = config.get('t_len')
learning_rate = config.get('learning_rate')
k_epochs = config.get('k_epochs')
eps_clip = config.get('eps_clip')
mini_batchs_size = config.get('mini_batchs_size')
concurrent_episodes = config.get('concurrent_episodes')
test_concurrent_episodes = config.get('test_concurrent_episodes')
episodes_rounds = config.get('episodes_rounds')
max_trade_steps = config.get('max_trade_steps')
cuda = config.get('cuda')
device = torch.device('cuda:0') if cuda else torch.device('cpu')
selected_net = 'backup.net'
load_selected_net = False

q_net = QNet(
    CoinNet(),
)

if load_selected_net:
    q_net.load_state_dict(torch.load(trading_agents_files_path + '/backup.net'))
    print('Net "'+selected_net+'" loaded.')

if cuda:
    q_net.cuda()

dqn_agent = DQN(
    q_net,
    learning_rate,
    calculate_gamma(horizon, t_len),
    k_epochs,
    eps_clip,
    ReplayBuffer(concurrent_episodes, (96, 14), cuda=cuda),
    soft_update_ratio,
    soft_update_period,
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
            RewardCalculator(spread=selected_instrument_spread, cuda=cuda),
            concurrent_episodes=concurrent_episodes,
            cuda=cuda
            )

test_gym = Gym(  selected_instrument,
                 granularity_list,
                 primary_granularity,
                 input_period_list,
                 test_from_datetime,
                 test_to_datetime,
                 episode_steps,
                 RewardCalculator(spread=selected_instrument_spread, cuda=cuda),
                 concurrent_episodes=test_concurrent_episodes,
                 cuda=cuda
                )

data_pre_processor = InputDataPreProcessor(
    concurrent_episodes,
    net_input_period_list[primary_granularity],
    recent_log_returns_length,
    cuda=cuda
)

test_data_pre_processor = InputDataPreProcessor(
    test_concurrent_episodes,
    net_input_period_list[primary_granularity],
    recent_log_returns_length,
    cuda=cuda
)

percentage = 50
percentage_threshold_list = [floor((i+1)*episode_steps/percentage) for i in range(percentage)]

def execute_episodes(gym, dqn_agent, data_pre_processor, testing=False):
    print(dqn_agent.t_steps)
    if testing:
        dqn_agent = copy.deepcopy(dqn_agent)

    done = False
    total_steps = 0
    percentage_now = 0

    data_pre_processor.reset()
    (candles_list, position_list) = gym.reset()

    name = 'testing' if testing else 'training'

    print('Executing '+name+' episodes [ ', end='', flush=True)
    while(not done):
        state = data_pre_processor.process(candles_list[0])
        # plot_tensor(state[0])
        action, q_values = dqn_agent.select_action(state)
        (candles_list, position), reward, done = gym.step(action)
        data_pre_processor.pushPosition(position)
        next_possible_states = data_pre_processor.patchAllPositions(candles_list[0])
        dqn_agent.buffer.push(state, next_possible_states, q_values, reward, action)
        dqn_agent.step()

        total_steps += 1
        if total_steps>=percentage_threshold_list[percentage_now]:
            percentage_now += 1
            print('>', end='', flush=True)

        # if total_steps%update_n_steps == 0 and dqn_agent.buffer.filled:
        # # #     # if testing:
        # # #     #     dqn_agent.clear()
        # # #     # else:
        # #     # print(dqn_agent.buffer.actions)
        #     action_rewards = torch.gather(dqn_agent.buffer.rewards, -1, dqn_agent.buffer.actions).squeeze(1)
        #     plot_tensor([dqn_agent.buffer.q_values.squeeze(1), data_pre_processor.toOneHot(dqn_agent.buffer.actions).squeeze(1), dqn_agent.buffer.rewards.squeeze(1), action_rewards, torch.cumsum(action_rewards, dim=0)], use_points=[1])
        #     dqn_agent.update()

    print(' ]')

    summary = gym.reward_calculator.get_summary()

    pretty_print(summary)

    dqn_agent.clear()

    return summary


reward_list = []
test_reward_list = []

log_returns_list = []
test_log_returns_list = []

for i in range(episodes_rounds):
    print('*** [Round '+str(i+1)+'] ***')

    summary = execute_episodes(gym, dqn_agent, data_pre_processor, False)
    reward_list.append(summary['total_reward_average'])
    log_returns_list.append(summary['total_log_returns_average'])

    # ************************************************************************testing

    # Save dqn_agent net cpu version
    if cuda:
        dqn_agent.policy_net.cpu()
        torch.cuda.empty_cache()

    torch.save(dqn_agent.policy_net.state_dict(), trading_agents_files_path + '/backup.net')

    if cuda:
        dqn_agent.policy_net.cuda()

    summary = execute_episodes(test_gym, dqn_agent, test_data_pre_processor, True)
    test_reward_list.append(summary['total_reward_average'])
    test_log_returns_list.append(summary['total_log_returns_average'])

    # Draw loss fig
    # print(reward_list)
    # print(test_reward_list)
    # print(log_returns_list)
    # print(test_log_returns_list)
    x = list(range(1, len(reward_list)+1))
    plt.plot(x, reward_list, label='Reward')
    plt.plot(x, test_reward_list, label='Test-Reward')
    plt.plot(x, log_returns_list, label='LogReturns')
    plt.plot(x, test_log_returns_list, label='Test-LogReturns')
    plt.legend()
    plt.savefig(trading_agents_files_path + '/' + '/reward.png')
    plt.clf()

if cuda:
    dqn_agent.policy_net.cpu()
    torch.cuda.empty_cache()

torch.save(dqn_agent.policy_net.state_dict(), trading_agents_files_path + '/' + datetime.now().strftime("/NT_%Y_%m_%d_%H_%M.net"))

  # "selected_instrument_list" : [  "eurusd", "gbpusd", "audusd", "eurjpy", "nzdusd",
  #                                 "usdcad", "usdjpy", "usdtry", "xagusd", "xauusd",
  #                                 "zarjpy", "usdzar", "usdczk", "usdnok", "usdsgd",
  #                                 "usdchf"
  #                             ],
  # 3840
  #
  # ["eurusd", "gbpusd", "audusd", "eurjpy", "nzdusd",
  #                             "usdcad", "usdjpy", "gbpaud", "euraud", "gbpcad",
  #                             "gbpnzd", "nzdchf", "cadchf", "eurcad", "gbpchf",
  #                             "audjpy", "eurnok", "usdtry", "audnzd", "audchf",
  #                             "sgdjpy", "xagusd", "xauusd", "zarjpy", "usdzar",
  #                             "gbpjpy", "usdczk", "audcad", "cadjpy", "chfjpy",
  #                             "eurgbp", "usdnok", "xauaud", "xaugbp", "xaueur",
  #                             "eurczk", "nzdcad", "usdsgd", "usdchf", "eurtry"
  #                           ]
