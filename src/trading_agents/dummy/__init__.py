#!/usr/bin/python3

from coin_wizard.historical_pair_data import get_historical_pair_data

class TradingAgent(object):
    def __init__(self, APIs):
        self.APIs = APIs

    def run(self, brokerAPI):
        print(self.APIs)
        self.APIs['test'](123)

    def stop_running():
        pass

    def train(self):
        get_historical_pair_data('eurusd')

    def stop_training():
        pass

    def test(self, brokerAPI):
        # print(self.APIs)
        self.APIs['test'](123)

    def stop_testing():
        pass
