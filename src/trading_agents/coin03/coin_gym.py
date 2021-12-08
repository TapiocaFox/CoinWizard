import torch
from data_loader import DataLoader
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

granularity_time_delta = {
    "M1": timedelta(seconds=60),
    "M5": timedelta(seconds=60*5),
    "M15": timedelta(seconds=60*15),
    "M30": timedelta(seconds=60*30),
    "H1": timedelta(seconds=60*60),
    "H4": timedelta(seconds=60*240),
    "D": timedelta(seconds=60*60*24),
}

class RealTimeGym(object):
    def __init__(self):
        pass


class Gym(object):
    def __init__(self, instrument, granularity_list, primary_granularity, input_period_list, from_datetime, to_datetime, episode_steps, reward_calculator, concurrent_episodes, cuda, verbose=True):
        self.verbose = verbose

        self.instrument = instrument
        self.granularity_list = granularity_list
        self.primary_granularity = primary_granularity
        self.input_period_list = input_period_list
        self.from_datetime = from_datetime
        self.to_datetime = to_datetime
        self.reward_calculator = reward_calculator
        self.episode_steps = episode_steps
        self.current_episode_steps = 0
        self.concurrent_episodes = concurrent_episodes
        self.episode_candles_fast_inference_index_list = []
        self.dataloader = DataLoader(instrument, granularity_list, primary_granularity, input_period_list, from_datetime, to_datetime, episode_steps+1, cuda=cuda)
        self.cuda = cuda
        self.device = torch.device('cuda:0') if cuda else torch.device('cpu')
        self.position_state_tensor = None
        self.short_position = 0
        self.wait_position = 1
        self.long_position = 2
        #
        # self.actions = ['short', 'wait', 'long']

    def _step_actions(self, actions):
        self.current_episode_steps += 1
        reward = self.reward_calculator.step(self.current_episode_steps, actions)
        self.position_state_tensor = actions

        return reward

    def _step_states(self):
        candles_states = [[None for i in range(self.concurrent_episodes)] for j in range(len(self.granularity_list))]

        # Concurrent episodes
        for episode_index, (candles_list, fast_inference_index_list) in enumerate(self.episode_candles_fast_inference_index_list):
            # Granularities
            for granularity, candles in enumerate(candles_list):
                hist_data = candles_list[granularity]
                input_period = self.input_period_list[granularity]
                granularity_random_index = fast_inference_index_list[self.current_episode_steps][granularity]
                candles_states[granularity][episode_index] = hist_data[granularity_random_index+1-input_period:granularity_random_index+1]

        candles_states = [torch.stack(candles_granularity) for candles_granularity in candles_states]

        return (candles_states, self.position_state_tensor)

    def reset(self):
        if self.verbose:
            print('Gym reset between date "' + str(self.from_datetime) + '" and "'+ str(self.to_datetime)+'". ')
            print('With selected instruments: '+str(self.instrument)+'.')

        del self.episode_candles_fast_inference_index_list[:]

        self.episode_candles_fast_inference_index_list = [self.dataloader.generateEpisode() for i in range(self.concurrent_episodes)]
        self.position_state_tensor = self.wait_position*torch.ones(self.concurrent_episodes, device=self.device)
        self.current_episode_steps = 0

        self.reward_calculator.reset(self.concurrent_episodes, self.granularity_list, self.episode_candles_fast_inference_index_list, self.primary_granularity, self.episode_steps)
        return self._step_states()

    def step(self, actions):
        reward = self._step_actions(actions)
        done = self.current_episode_steps >= self.episode_steps
        return self._step_states(), reward, done
