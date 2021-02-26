#!/usr/bin/python3

import plotly.graph_objects as go

from datetime import datetime
from coin_wizard.historical_pair_data import get_historical_pair_data_pandas
# import state manager

class TradingAgent(object):
    def __init__(self, agent_directory):
        print(agent_directory)

    def run(self, brokerAPI):
        pass

    def stop_running():
        pass

    def train(self):
        quotes = get_historical_pair_data_pandas('eurusd', datetime(2018, 5, 17), datetime(2021, 1, 17))
        print(quotes)
        # fig = go.Figure(data=[go.Candlestick(x=quotes['f0'],
        #         open=quotes['f1'],
        #         high=quotes['f2'],
        #         low=quotes['f3'],
        #         close=quotes['f4'])])

    def stop_training():
        pass

    def test(self, BacktestBrokerAPI):
        # BacktestBrokerAPI.start()
        # print(self.APIs)
        pass

    def stop_testing():
        pass
