#!/usr/bin/python3

class TradingAgent(object):
    def __init__(self, APIs):
        self.APIs = APIs

    def run(self):
        # print(self.APIs)
        self.APIs['test'](123)

    def train(self):
        # print(self.APIs)
        self.APIs['test'](123)

    def test(self):
        # print(self.APIs)
        self.APIs['test'](123)
