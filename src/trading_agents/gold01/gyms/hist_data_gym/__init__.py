from ..gym import Gym
from .data_loader import DataLoader

class HistDataGym(Gym):
    def __init__(self, instrument_list, primary_intrument, granularity, observation_period, from_datetime, to_datetime, episode_steps, reward_calculator, account_book):
        self.on_step_callback = None
        self.instrument_list = instrument_list
        self.primary_intrument = primary_intrument
        self.granularity = granularity
        self.observation_period = observation_period
        self.data_loader = DataLoader(instrument_list, primary_intrument, granularity, observation_period, from_datetime, to_datetime, episode_steps+1)
        self.episode_steps = episode_steps
        self.reward_calculator = reward_calculator
        self.account_book = account_book

    def generateOutSpecs(self):
        return {
            'instrument_list': self.instrument_list,
            'primary_intrument': self.primary_intrument,
            'granularity': self.granularity,
            'observation_period': self.observation_period,
            'open_prefix': 'open', 'high_prefix': 'high', 'low_prefix': 'low', 'close_prefix': 'close'
        }

    def start(self):
        episode_steps = self.episode_steps
        observation_period = self.observation_period
        hist_data, first_index = self.data_loader.generateEpisode()
        self.reward_calculator.reset(self.generateOutSpecs())

        for i in range(episode_steps):
            # Plus one since "latest known" index.
            # Before action taken.
            latest_known_index = first_index+i
            observed_hist_data = hist_data.iloc[(latest_known_index+1)-observation_period:(latest_known_index+1)].reset_index()
            finished = i==episode_steps-1
            action = self.on_step_callback(finished, observed_hist_data)

            # After action taken. "Plus two".
            next_known_hist_data = hist_data.iloc[:(latest_known_index+2)]
            reward, trade = self.reward_calculator.calculateReward(next_known_hist_data, action)
            if trade is not None:
                self.account_book.appendTrade(trade)

        raise NotImplementedError('Not implemented.')

    def onStep(self, callback):
        self.on_step_callback = callback
        # action = callback(finished, state)
        # raise NotImplementedError('Not implemented.')

    def onFinished(self, callback):
        self.on_finished_callback = callback
