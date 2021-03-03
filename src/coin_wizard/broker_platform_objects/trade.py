#!/usr/bin/python3
#

class Trade(object):
    def __init__(self, trade_id, instrument_name, open_price, trade_settings, update_trade):
        self.trade_id = trade_id
        self.instrument_name = instrument_name
        self.open_price = open_price
        self.trade_settings = trade_settings

        self.price = open_price
        self.unrealized_pl = 0
        self.closed_listener = None
        self.closed = False
        self.update_trade = update_trade

    def close(self):
        if self.closed:
            raise Exception('Trade already closed.')
        return self.close_handler(self.trade_id)

    def modify(self, trade_settings):
        if self.closed:
            raise Exception('Trade already closed.')
        return self.modify_handler(self.trade_id, trade_settings)

    def getInstrumentName(self):
        return self.instrument_name

    def getOpenPrice(self):
        return self.open_price

    def getTradeSettings(self):
        return self.trade_settings

    def getUnrealizedPL(self):
        self.update_trade(self)
        return self.unrealized_pl

    def onClosed(self, closed_listener):
        self.closed_listener = closed_listener
