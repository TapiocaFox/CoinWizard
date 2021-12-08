from utils import ConfigLoader
from data_loader import DataLoader
from coin_gym import Gym
from reward import RewardCalculator
from utils import InputDataPreProcessor, calculate_gamma, plot_list, pretty_print, plot_tensor
from dqn import DQN, QNet, ReplayBuffer
from ppo import PPO, ActorCritic, ReplayBuffer as PPOReplayBuffer
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

selected_instrument_list = config.get('selected_instrument_list')
primary_intrument = config.get('primary_intrument')
primary_instrument_spread = config.get('primary_instrument_spread')
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

dqn_or_ppo_mode = config.get('dqn_or_ppo_mode')
episode_steps = config.get('episode_steps')
update_n_steps = config.get('update_n_steps')
soft_update_ratio = config.get('soft_update_ratio')
soft_update_period = config.get('soft_update_period')
horizon = config.get('horizon')
t_len = config.get('t_len')
learning_rate = config.get('learning_rate')
ppo_learning_rate = config.get('ppo_learning_rate')
k_epochs = config.get('k_epochs')
eps_clip = config.get('eps_clip')
mini_batchs_size = config.get('mini_batchs_size')
concurrent_episodes = config.get('concurrent_episodes')
test_concurrent_episodes = config.get('test_concurrent_episodes')
episodes_rounds = config.get('episodes_rounds')
cuda = config.get('cuda')
device = torch.device('cuda:0') if cuda else torch.device('cpu')
selected_net = 'backup.net'
load_selected_net = False

if dqn_or_ppo_mode:
    actor_critic = ActorCritic(
        CoinNet(102, critic=False),
        CoinNet(102, critic=True),
    )

    if load_selected_net:
        actor_critic.load_state_dict(torch.load(trading_agents_files_path + '/backup.net'))
        print('Net "'+selected_net+'" loaded.')
    else:
        actor_critic.actor.initialize()
        actor_critic.critic.initialize()

    if cuda:
        actor_critic.cuda()

    ppo_agent = PPO(
        actor_critic,
        ppo_learning_rate,
        ppo_learning_rate,
        calculate_gamma(horizon, t_len),
        soft_update_ratio,
        soft_update_period,
        k_epochs,
        eps_clip,
        PPOReplayBuffer(concurrent_episodes, (96, 102), cuda=cuda),
        mini_batchs_size,
        cuda=cuda
    )

else:
    q_net = QNet(
        CoinNet(102, q_net=True)
    )

    if load_selected_net:
        q_net.load_state_dict(torch.load(trading_agents_files_path + '/backup.net'))
        print('Net "'+selected_net+'" loaded.')
    else:
        q_net.model.initialize()

    if cuda:
        q_net.cuda()

    dqn_agent = DQN(
        q_net,
        learning_rate,
        calculate_gamma(horizon, t_len),
        k_epochs,
        eps_clip,
        ReplayBuffer(concurrent_episodes, (96, 102), cuda=cuda),
        soft_update_ratio,
        soft_update_period,
        mini_batchs_size,
        cuda=cuda
    )

gym = Gym(  selected_instrument_list,
            primary_intrument,
            granularity_list,
            primary_granularity,
            input_period_list,
            from_datetime,
            to_datetime,
            episode_steps,
            RewardCalculator(spread=primary_instrument_spread, cuda=cuda),
            concurrent_episodes=concurrent_episodes,
            cuda=cuda
            )

test_gym = Gym(  selected_instrument_list,
                 primary_intrument,
                 granularity_list,
                 primary_granularity,
                 input_period_list,
                 test_from_datetime,
                 test_to_datetime,
                 episode_steps,
                 RewardCalculator(spread=primary_instrument_spread, cuda=cuda),
                 concurrent_episodes=test_concurrent_episodes,
                 cuda=cuda
                )

data_pre_processor = InputDataPreProcessor(
    concurrent_episodes,
    selected_instrument_list,
    primary_intrument,
    net_input_period_list[primary_granularity],
    recent_log_returns_length,
    cuda=cuda
)

test_data_pre_processor = InputDataPreProcessor(
    test_concurrent_episodes,
    selected_instrument_list,
    primary_intrument,
    net_input_period_list[primary_granularity],
    recent_log_returns_length,
    cuda=cuda
)

percentage = 50
percentage_threshold_list = [floor((i+1)*episode_steps/percentage) for i in range(percentage)]

def execute_episodes(gym, agent, data_pre_processor, testing=False):
    print(agent.t_steps)
    if testing:
        agent = copy.deepcopy(agent)

    done = False
    total_steps = 0
    percentage_now = 0

    data_pre_processor.reset()
    (candles_list, position_list) = gym.reset()

    name = 'testing' if testing else 'training'

    print('Executing '+name+' episodes [ ', end='', flush=True)
    while(not done):
        state = data_pre_processor.process(candles_list[0])
        # print(state[0])
        # plot_tensor([state[0]])
        if dqn_or_ppo_mode:
            actions, policy_probs = agent.select_action(state)
            policy_probs, states_values, _ = agent.policy_old.evaluate(state)
            (candles_list, position), reward, done = gym.step(actions)
            data_pre_processor.pushPosition(position)
            next_possible_states = data_pre_processor.patchAllPositions(candles_list[0])
            next_possible_states_values = agent.policy_old.evaluate_next_states(next_possible_states)
            agent.buffer.push(state, next_possible_states, reward, actions, policy_probs, states_values, next_possible_states_values)
            agent.step()

        else:
            action, q_values = agent.select_action(state)
            (candles_list, position), reward, done = gym.step(action)
            data_pre_processor.pushPosition(position)
            next_possible_states = data_pre_processor.patchAllPositions(candles_list[0])
            agent.buffer.push(state, next_possible_states, q_values, reward, action)
            agent.step()

        total_steps += 1
        if total_steps>=percentage_threshold_list[percentage_now]:
            percentage_now += 1
            print('>', end='', flush=True)

        if total_steps%update_n_steps == 0 and agent.buffer.filled:
        # # #     # if testing:
        # # #     #     dqn_agent.clear()
        # # #     # else:
        # #     # print(dqn_agent.buffer.actions)
            buffer = agent.buffer
            action_rewards = torch.gather(buffer.rewards, -1, buffer.actions).squeeze(1)
            # plot_tensor([dqn_agent.buffer.q_values.squeeze(1), data_pre_processor.toOneHot(dqn_agent.buffer.actions).squeeze(1), dqn_agent.buffer.rewards.squeeze(1), action_rewards, torch.cumsum(action_rewards, dim=0)], use_points=[1])
            # plot_tensor([buffer.policy_probs.squeeze(1), buffer.states_values.squeeze(1), data_pre_processor.toOneHot(buffer.actions).squeeze(1), buffer.rewards.squeeze(1), action_rewards, torch.cumsum(action_rewards, dim=0)], use_points=[0, 1, 2])
        #     agent.update()

    print(' ]')

    summary = gym.reward_calculator.get_summary()

    pretty_print(summary)

    agent.clear()

    return summary


reward_list = []
test_reward_list = []

reward_without_spread_list = []
test_reward_without_spread_list = []

for i in range(episodes_rounds):
    print('*** [Round '+str(i+1)+'] ***')

    if dqn_or_ppo_mode:
        summary = execute_episodes(gym, ppo_agent, data_pre_processor, False)
    else:
        summary = execute_episodes(gym, dqn_agent, data_pre_processor, False)

    reward_list.append(summary['total_reward_average'])
    reward_without_spread_list.append(summary['total_reward_without_spread_tensor'])

    # ************************************************************************testing

    # Save dqn_agent net cpu version
    if dqn_or_ppo_mode:
        if cuda:
            ppo_agent.policy.cpu()
            torch.cuda.empty_cache()
        torch.save(ppo_agent.policy.state_dict(), trading_agents_files_path + '/backup.net')
        if cuda:
            ppo_agent.policy.cuda()
    else:
        if cuda:
            dqn_agent.policy_net.cpu()
            torch.cuda.empty_cache()
        torch.save(dqn_agent.policy_net.state_dict(), trading_agents_files_path + '/backup.net')
        if cuda:
            dqn_agent.policy_net.cuda()


    if dqn_or_ppo_mode:
        summary = execute_episodes(gym, ppo_agent, data_pre_processor, True)
    else:
        summary = execute_episodes(gym, dqn_agent, data_pre_processor, True)

    test_reward_list.append(summary['total_reward_average'])
    test_reward_without_spread_list.append(summary['total_reward_without_spread_tensor'])

    # Draw loss fig
    # print(reward_list)
    # print(test_reward_list)
    # print(reward_without_spread_list)
    # print(test_reward_without_spread_list)
    x = list(range(1, len(reward_list)+1))
    plt.plot(x, reward_list, label='Reward')
    plt.plot(x, test_reward_list, label='Test-Reward')
    plt.plot(x, reward_without_spread_list, label='RewardWithoutSpread')
    plt.plot(x, test_reward_without_spread_list, label='Test-RewardWithoutSpread')
    plt.legend()
    plt.savefig(trading_agents_files_path + '/' + '/reward.png')
    plt.clf()


if dqn_or_ppo_mode:
    if cuda:
        ppo_agent.policy.cpu()
        torch.cuda.empty_cache()
    torch.save(ppo_agent.policy.state_dict(), trading_agents_files_path + '/' + datetime.now().strftime("/NT_%Y_%m_%d_%H_%M.net"))
else:
    if cuda:
        dqn_agent.policy_net.cpu()
        torch.cuda.empty_cache()
    torch.save(dqn_agent.policy_net.state_dict(), trading_agents_files_path + '/' + datetime.now().strftime("/NT_%Y_%m_%d_%H_%M.net"))
