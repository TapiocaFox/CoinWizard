#!/usr/bin/python3
#

class Order(object):
    def __init__(self, order_id, instrument_name, order_settings, trade_settings):
        self.order_id = order_id
        self.instrument_name = instrument_name
        self.order_settings = order_settings
        self.trade_settings = trade_settings
        self.filled_listener = None
        self.canceled_listener = None
        self.cancel_handler = None

    def getInstrumentName(self):
        return self.instrument_name

    def getOrderSettings(self):
        return self.order_settings

    def getTradeSettings(self):
        return self.trade_settings

    def cancel(self):
        return self.cancel_handler(self.order_id)

    def onFilled(self, filled_listener):
        self.filled_listener = filled_listener

    def onCanceled(self, canceled_listener):
        self.canceled_listener = canceled_listener
