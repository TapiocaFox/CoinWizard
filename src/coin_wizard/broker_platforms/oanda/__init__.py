#!/usr/bin/python3

import json

from oandapyV20 import API

import coin_wizard.broker_platform_objects as BrokerPlatform
from time import sleep

import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.transactions as transactions

class BrokerEventLoopAPI(BrokerPlatform.BrokerEventLoopAPI):
    hedging = False
    broker_settings_fields = ['access_token', 'account_id']
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms = 50):
        super().__init__(before_loop, after_loop, broker_settings, loop_interval_ms)
        self.oanda_api = API(access_token=broker_settings['access_token'])
        self.account_id = broker_settings['account_id']
        self.account = BrokerPlatform.Account()

        # Initializing
        # Acount
        # r = transactions.TransactionIDRange(self.account_id, {"from": 1, "to": 30})
        # rv = self.oanda_api.request(r)
        # print(json.dumps(rv, indent=2))
        # Acount
        r = accounts.AccountDetails(self.account_id)
        rv = self.oanda_api.request(r)

        account = rv['account']
        self.balance = account['balance']
        self.currency = account['currency'].lower()
        self.margin_rate = account['marginRate']
        self.margin_used = account['marginUsed']
        self.margin_available = account['marginAvailable']
        self.unrealized_pl = account['unrealizedPL']

        for order_detail in account['orders']:
            if order_detail['type'] in ['MARKET', 'LIMIT', 'STOP']:
                order_settings = {
                    "price": order_detail['price']
                }

                trade_settings = {}

                if "units" in order_detail:
                    trade_settings['units'] = order_detail['units']

                if "takeProfitOnFill" in order_detail:
                    trade_settings['take_profit'] = order_detail['takeProfitOnFill']['price']

                if "stopLossOnFill" in order_detail:
                    trade_settings['stop_lost'] = order_detail['stopLossOnFill']['price']

                if "trailingStopLossOnFill" in order_detail:
                    trade_settings['trailing_stop_distance'] = order_detail['trailingStopLossOnFill']['distance']

                # print(json.dumps(order_detail, indent=2))

                order = BrokerPlatform.Order(order_detail['id'], order_detail['instrument'], order_settings, trade_settings)
                order.cancel_handler = self._order_cancel_handler
                # order
                self.account.orders.append(order)

        for trade_detail in account['trades']:

            trade_settings = {}

            trade_settings['units'] = trade_detail['currentUnits']
            # trade_settings['open_price'] = trade_detail['price']

            if "takeProfitOrderID" in trade_detail:
                r = orders.OrderDetails(self.account_id, trade_detail['takeProfitOrderID'])
                rv = self.oanda_api.request(r)
                trade_settings['take_profit'] = rv['order']['price']

            if "stopLossOrderID" in trade_detail:
                r = orders.OrderDetails(self.account_id, trade_detail['stopLossOrderID'])
                rv = self.oanda_api.request(r)
                trade_settings['stop_loss'] = rv['order']['price']

            if "trailingStopLossOrderID" in trade_detail:
                r = orders.OrderDetails(self.account_id, trade_detail['trailingStopLossOrderID'])
                rv = self.oanda_api.request(r)
                trade_settings['trailing_stop_distance'] = rv['order']['distance']

            # print(json.dumps(trade_detail, indent=2))

            trade = BrokerPlatform.Trade(trade_detail['id'], trade_detail['instrument'], trade_detail['price'], trade_settings)

            if "takeProfitOrderID" in trade_detail:
                trade.take_profit_order_id = trade_detail['takeProfitOrderID']
            if "stopLossOrderID" in trade_detail:
                trade.stop_lost_order_id = trade_detail['stopLossOrderID']
            if "trailingStopLossOrderID" in trade_detail:
                trade.trailing_stop_order_id = trade_detail['trailingStopLossOrderID']

            trade.close_handler = None
            trade.modify_handler = None

            self.account.trades.append(trade)


        # print(json.dumps(rv, indent=2))
        # Order
        # r = orders.OrderList(self.account_id)
        # rv = self.oanda_api.request(r)
        # print(json.dumps(rv, indent=2))
        # Trades
        # r = trades.TradesList(self.account_id)
        # rv = self.oanda_api.request(r)
        # print(json.dumps(rv, indent=2))

    def order(self, order_settings, trade_settings):
        pass

    def getInstrument(self, instrument_name):
        pass

    def _order_cancel_handler(self, order_id):
        pass

    def _trade_close_handler(self, trade_id):
        pass

    def _trade_modify_handler(self, trade_id, trade_settings):
        pass

    def _loop(self):
        pass
