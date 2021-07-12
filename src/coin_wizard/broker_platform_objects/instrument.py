#!/usr/bin/python3
#

class Instrument(object):
    def __init__(self, instrument_name, update_instrument):
        self.instrument_name = instrument_name
        self.changed_listener = None
        self.recent_candles = {
            'M1': None,
            'M5': None,
            'M15': None,
            'M30': None,
            'H1': None,
            'H4': None,
            'D': None,
        }
        self.active_candle = {
            'M1': None,
            'M5': None,
            'M15': None,
            'M30': None,
            'H1': None,
            'H4': None,
            'D': None,
        }
        self.active_1m_candle = None
        self.update_instrument = update_instrument
        self.tradable = True
        self.current_closeout_bid = 0.0
        self.current_closeout_ask = 0.0
        self.current_closeout_bid_ask_datetime = None

    def getActive1MCandle(self):
        return self.getActiveCandle('M1')

    def getActiveCandle(self, granularity='M1'):
        self.update_instrument(self)
        return self.active_candle[granularity]

    def getCurrentCloseoutBidAsk(self):
        return self.current_closeout_bid, self.current_closeout_ask, self.current_closeout_bid_ask_datetime

    def getRecent1MCandles(self, counts=500):
        return self.getRecentCandles(counts, 'M1')

    def getRecentCandles(self, counts=500, granularity='M1'):
        self.update_instrument(self)
        return self.recent_candles[granularity].tail(counts).reset_index(drop=True).copy()

    def isTradable(self):
        self.update_instrument(self)
        return self.tradable
