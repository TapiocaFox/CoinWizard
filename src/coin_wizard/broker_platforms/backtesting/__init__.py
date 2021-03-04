#!/usr/bin/python3

import json, dateutil
import pandas as pd

import coin_wizard.broker_platform_objects as BrokerPlatform
from datetime import datetime
from time import sleep

from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.transactions as transactions

from oandapyV20.contrib.requests import MarketOrderRequest

update_interval_threshold_ms = 50

class BrokerEventLoopAPI(BrokerPlatform.BrokerEventLoopAPI):
    hedging = False
    broker_settings_fields = ['access_token', 'account_id']
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms = 50):
        super().__init__(before_loop, after_loop, broker_settings, loop_interval_ms)
        self.instruments_watchlist = {}
        self.virtual_datetime = 0
        self.account = BrokerPlatform.Account(self._update_account_handler)

    def order(self, instrument_name, order_settings, trade_settings):

        order_settings_r = {
            "instrument": instrument_name,
            "type": order_settings['type'].upper(),
            "price": "2",
        }

        if "bound" in order_settings:
            order_settings_r['priceBound'] = order_settings['bound']

        # trade_settings
        if "units" in trade_settings:
            order_settings_r['units'] = trade_settings['units']

        if "take_profit" in trade_settings:
            order_settings_r['takeProfitOnFill'] = {
                "price": trade_settings['take_profit']
            }

        if "stop_lost" in trade_settings:
            order_settings_r['stopLossOnFill'] = {
                "price": trade_settings['stop_lost']
            }

        if "trailing_stop_distance" in trade_settings:
            order_settings_r['trailingStopLossOnFill'] = {
                "distance": trade_settings['trailing_stop_distance']
            }

        rv['orderCreateTransaction']['type'] = order_settings['type'].upper()
        return

    def getInstrument(self, instrument_name):
        if instrument_name in self.instruments_watchlist:
            return self.instruments_watchlist[instrument_name]

        instrument = BrokerPlatform.Instrument(instrument_name, self._update_instrument_handler)
        return instrument

    def _order_cancel_handler(self, order_id):
        pass

    def _trade_close_handler(self, trade_id):
        pass

    def _trade_modify_handler(self, trade_id, trade_settings):
        pass

    def _update_instrument_handler(self, instrument):
        pass

    def _update_account_handler(self):
        pass

    def _update_trade_handler(self, trade):
        pass

    def _loop(self):
        pass

    # For Neural Net
    def resetByDate(self, start_datetime, end_datetime):
        pass
