from .reward_calculator import RewardCalculator

class RealisticRewardCalculator(RewardCalculator):
    def __init__(self):
        return

    def reset(self):
        raise NotImplementedError('Not implemented.')

    def calculateReward(self, known_ohlc_df, action):
        print(known_ohlc_df)

        return None, None
