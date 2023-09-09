from .reward_calculator import RewardCalculator
import math

class EveryStepRewardCalculator(RewardCalculator):
    def __init__(self, spread):
        self.spread = spread

    def reset(self, gym_out_specs):
        self.previous_position = 0
        # self.trade_open_price = None
        self.instrument_list = gym_out_specs['instrument_list']
        self.primary_intrument = gym_out_specs['primary_intrument']
        self.primary_close_label = gym_out_specs['close_prefix']+str(self.primary_intrument)
        self.primary_open_label = gym_out_specs['open_prefix']+str(self.primary_intrument)
        self.trade_open_index = None

    def calculateReward(self, next_known_ohlc_df, action):
        finished_trade = None

        half_spread = abs(action-self.previous_position)*self.spread
        next_candle = next_known_ohlc_df.iloc[-1]
        latest_known_candle = next_known_ohlc_df.iloc[-2]

        price_diff = next_candle[self.primary_close_label]-latest_known_candle[self.primary_close_label]
        step_reward = action*price_diff
        reward = step_reward-half_spread

        # Close trade
        if self.previous_position != 0 and (action != self.previous_position):
            finished_trade = {
                'position': self.previous_position,
                'spread': self.spread,
                'instrument': self.instrument_list[self.primary_intrument],
                'dataframe': next_known_ohlc_df.iloc[self.trade_open_index:]
            }
            self.trade_open_index = None

        # New trade
        if action != 0 and (action != self.previous_position):
            self.trade_open_index = next_known_ohlc_df.iloc[-1].name

        # Set new previous_position
        self.previous_position = action

        return reward, finished_trade
