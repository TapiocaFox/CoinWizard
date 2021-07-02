#!/usr/bin/python3

class TechnicalIndicators(object):
    def __init__(self):
        pass

    # Moving average
    def ma(self, series, period=10):
        return series.rolling(window=period, min_periods=period).mean()

    # Exponential moving average
    def ema(self, series, period=10):
        return series.ewm(span=period, min_periods=period).mean()

    # Moving average convergence divergence
    def macd(self, series, short=12, long=26):
        return series.ewm(span=short).mean() - series.ewm(span=long).mean()

    # Rate of change
    def roc(self, series, period=2):
        return series.diff(period-1)/series.shift(period-1)

    # Momentum
    def momentum(self, series, period=4):
        return series.diff(period)

    # Relative strength index ma
    def rsi_ma(self, series, period=10):
        delta = series.diff()
        # delta = delta[1:]

        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        roll_up = up.rolling(period).mean()
        roll_down = down.abs().rolling(period).mean()

        rs = roll_up / roll_down
        return 100.0 - (100.0 / (1.0 + rs))

    # Relative strength index ema
    def rsi_ema(self, series, period=10):
        delta = series.diff()
        # delta = delta[1:]

        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        roll_up = up.ewm(span=period).mean()
        roll_down = down.abs().ewm(span=period).mean()

        rs = roll_up / roll_down
        return 100.0 - (100.0 / (1.0 + rs))

    # Bollinger bands
    def bb(self, series, period=20, std_num=2):
        rolling_mean = series.rolling(window=period).mean()
        rolling_std  = series.rolling(window=period).std()
        return rolling_mean + (rolling_std*std_num), rolling_mean - (rolling_std*std_num)

    # Commodity channel index
    def cci(self, high_series, low_series, close_series, period=20):
        pp = (high_series + low_series + close_series) / 3
        return (pp - pp.rolling(period, min_periods=period).mean()) / (0.015 * pp.rolling(period, min_periods=period).std())

    # # Yeh trapzoid index
    # def yti(self):
    #     pass
