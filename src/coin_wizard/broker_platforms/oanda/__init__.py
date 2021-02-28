#!/usr/bin/python3

# import json

from oandapyV20 import API

import coin_wizard.broker_platform_objects as BrokerPlatform
from time import sleep

import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing

class BrokerEventLoopAPI(BrokerPlatform.BrokerEventLoopAPI):
    broker_settings_fields = ['access_token', 'account_id']
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms= 50):
        super().__init__(before_loop, after_loop, broker_settings, loop_interval_ms)
        self.oanda_api = API(access_token=broker_settings['access_token'])
        self.account_id = broker_settings['account_id']
        self.account = BrokerPlatform.Account()
        # Initializing
        # Order
        r = orders.OrderList(self.account_id)
        rv = self.oanda_api.request(r)
        print(rv)
        # Trades
        r = trades.TradesList(self.account_id)
        rv = self.oanda_api.request(r)
        print(rv)

    def order(self, order_settings, trade_settings):
        pass

    def getInstrument(self, instrument_name):
        pass

    def getAccount(self):
        pass

    def __loop(self):
        pass
