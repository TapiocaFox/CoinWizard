#!/usr/bin/python3

import threading

class TradingAgent(threading.Thread):
    def __init__(self, APIs):
        threading.Thread.__init__(self)
        self.APIs = APIs
        self.APIs['on']('CliMessage', self.CliMessageListener)

    def CliMessageListener(self, message):
        print(message)

    def run(self):
        # print(self.APIs)
        self.APIs['test'](123)
