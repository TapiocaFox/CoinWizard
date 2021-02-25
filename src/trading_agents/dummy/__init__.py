#!/usr/bin/python3

from coin_wizard.historical_pair_data import get_historical_pair_data

class TradingAgent(object):
    def __init__(self, agent_directory):
        print(agent_directory)

    def run(self, brokerAPI):
        pass

    def stop_running():
        pass

    def train(self):
        get_historical_pair_data('eurusd')

    def stop_training():
        pass

    def test(self, BacktestBrokerAPI):
        # BacktestBrokerAPI.start()
        # print(self.APIs)
        pass

    def stop_testing():
        pass
