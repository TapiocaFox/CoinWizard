#!/usr/bin/python3

import plotly.graph_objects as go
import pytz
eastern = pytz.timezone('US/Eastern')
utc = pytz.utc

from datetime import datetime
from coin_wizard.historical_pair_data import plot_historical_pair_data
# import state manager

class TradingAgent(object):
    def __init__(self, agent_directory):
        print(agent_directory)

    def run(self, brokerAPI):
        account = brokerAPI.getAccount()
        orders = account.getOrders()
        trades = account.getTrades()

        for order in orders:
            print(order.getInstrumentName(), order.getOrderSettings(), order.getTradeSettings())

        for trade in trades:
            print(trade.getInstrumentName(), trade.getTradeSettings())

    def stop_running():
        pass

    def train(self):
        plot_historical_pair_data('eurusd', eastern.localize(datetime(2021, 1, 8, 0, 0)), eastern.localize(datetime(2021, 1, 11, 23, 59)), 'US/Eastern')

    def stop_training():
        pass

    def test(self, BacktestBrokerAPI):
        # BacktestBrokerAPI.start()
        # print(self.APIs)
        pass

    def stop_testing():
        pass
