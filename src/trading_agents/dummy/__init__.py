#!/usr/bin/python3

from datetime import datetime
from coin_wizard.historical_pair_data import get_historical_pair_data
# import state manager

class TradingAgent(object):
    def __init__(self, agent_directory):
        print(agent_directory)

    def run(self, brokerAPI):
        pass

    def stop_running():
        pass

    def train(self):
        print(get_historical_pair_data('eurusd', datetime(2002, 5, 17), datetime(2021, 1, 17)))

    def stop_training():
        pass

    def test(self, BacktestBrokerAPI):
        # BacktestBrokerAPI.start()
        # print(self.APIs)
        pass

    def stop_testing():
        pass
