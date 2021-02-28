#!/usr/bin/python3
#

class Trade(object):
    def __init__(self):
        self.onclosed_listener = None

    def close(self):
        pass

    def modify(self, trade_settings):
        pass

    def onClosed(self, onclosed_listener):
        self.onclosed_listener = onclosed_listener
