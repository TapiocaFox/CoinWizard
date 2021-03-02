#!/usr/bin/python3
#

class Instrument(object):
    def __init__(self):
        self.changed_listener = None

    def getRecentCandles1M(self, counts=500):
        pass
        
    def isTradable(self):
        pass

    def onPriceChanged(self, changed_listener):
        self.changed_listener = changed_listener
