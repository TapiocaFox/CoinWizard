#!/usr/bin/python3

from oandapyV20 import API

import coin_wizard.broker_platform_object as BrokerPlatform
from time import sleep
from coin_wizard.broker_platforms.backtest.supported_instruments import SupportedInstruments

class BrokerEventLoopAPI(BrokerPlatform.BrokerEventLoopAPI):
    SupportedInstruments = SupportedInstruments
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms= 50):
        super().__init__(self, before_loop, after_loop, broker_settings, loop_interval_ms)
        self.oanda_api = API(access_token=broker_settings.access_token)
        self.account_id = broker_settings.account_id
        self.orders = []
        self.trades = []
        # Initializing

    def order(self, order_settings, trade_settings):
        pass

    def getInstrument(self, instrument_name):
        pass

    def getAccount(self):
        pass

    def __loop(self):
        pass
