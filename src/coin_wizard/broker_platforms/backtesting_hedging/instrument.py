#!/usr/bin/python3

class Instrument(object):
    def onPriceChanged(self, changed_listener):
        pass

    def getRecentCandles1M(self, counts=500):
        pass

    # For Neural Net
    def foreseeFutureCandles1M(self, counts=50):
        pass
