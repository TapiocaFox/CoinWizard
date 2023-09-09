
class RewardCalculator:
    def __init__(self):
        return

    def reset(self):
        raise NotImplementedError('Not implemented.')

    def calculateReward(self, known_ohlc_df, action):
        raise NotImplementedError('Not implemented.')

        # return reward, trade_df
