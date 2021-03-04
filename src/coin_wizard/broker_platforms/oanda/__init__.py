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
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms = 1000):
        super().__init__(before_loop, after_loop, broker_settings, loop_interval_ms)
        self.oanda_api = API(access_token=broker_settings['access_token'])
        self.account_id = broker_settings['account_id']
        self.instruments_watchlist = {}

        self.latest_sync_transaction_id = 0

        self.account = BrokerPlatform.Account(self._update_account_handler)
        self.account.latest_update_datetime = datetime.now()

        # Initializing
        # Account
        r = accounts.AccountDetails(self.account_id)
        rv = self.oanda_api.request(r)
        # print(json.dumps(rv, indent=2))

        account = rv['account']
        self.latest_sync_transaction_id = int(account['lastTransactionID'])
        self.account.balance = account['balance']
        self.account.currency = account['currency']
        self.account.margin_rate = account['marginRate']
        self.account.margin_used = account['marginUsed']
        self.account.margin_available = account['marginAvailable']
        self.account.unrealized_pl = account['unrealizedPL']

        for order_detail in account['orders']:
            try:
                # print(json.dumps(order_detail, indent=2))
                self._import_order_detail(order_detail)
                # print(order_detail)
            except Exception as e:
                # print(json.dumps(order_detail, indent=2))
                # print(e)
                pass

        for trade_detail in account['trades']:
            # print(json.dumps(trade_detail, indent=2))
            self._import_trade_detail(trade_detail)

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

        r = orders.OrderCreate(self.account_id, {'order': order_settings_r})
        rv = self.oanda_api.request(r)
        # print(json.dumps(rv, indent=2))

        rv['orderCreateTransaction']['type'] = order_settings['type'].upper()
        return self._import_order_detail(rv['orderCreateTransaction'])

    def getInstrument(self, instrument_name):
        if instrument_name in self.instruments_watchlist:
            return self.instruments_watchlist[instrument_name]
        params = {
            "granularity": "M1",
            "count": 2000,
            # "count": 3,
        }

        instrument = BrokerPlatform.Instrument(instrument_name, self._update_instrument_handler)

        r = instruments.InstrumentsCandles(instrument_name, params)
        rv = self.oanda_api.request(r)
        candles = rv['candles']

        instrument.latest_candles_iso_time = candles[-1]['time']
        candles_df = self._convert_mid_candles_to_dataframe(candles)
        instrument.recent_1m_candles = candles_df.loc[candles_df['completed'] == True]
        instrument.latest_update_datetime = datetime.now()
        # instrument.pricing_stream = pricing.PricingStream(accountID=self.account_id, params={"instruments":instrument_name})
        # print([R for R in self.oanda_api.request(instrument.pricing_stream)])
        # raise
        # instrument.current_price =
        # print(instrument.recent_1m_candles)
        # print(instrument.recent_1m_candles.tail(1))
        self.instruments_watchlist[instrument_name] = instrument
        # raise
        return instrument

    def _import_order_detail(self, order_detail):
        if order_detail['type'] in ['MARKET', 'LIMIT', 'STOP']:
            order_settings = {
                "type": order_detail['type'].lower()
            }

            if "price" in order_detail:
                order_settings['price'] = order_detail['price']

            if "priceBound" in order_detail:
                order_settings['bound'] = order_detail['priceBound']

            trade_settings = {}

            if "units" in order_detail:
                trade_settings['units'] = float(order_detail['units'])

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
            return order
        else:
            # print(order_detail)
            raise Exception('Cannot import order detail.')

    def _import_trade_detail(self, trade_detail):
        # print(json.dumps(trade_detail, indent=2))
        trade_settings = {}

        trade_settings['units'] = float(trade_detail['currentUnits'])

        if "takeProfitOrder" in trade_detail:
            trade_settings['take_profit'] = float(trade_detail['takeProfitOrder']['price'])

        elif "takeProfitOrderID" in trade_detail:
            r = orders.OrderDetails(self.account_id, trade_detail['takeProfitOrderID'])
            rv = self.oanda_api.request(r)
            trade_settings['take_profit'] = float(rv['order']['price'])

        if "stopLossOrder" in trade_detail:
            trade_settings['stop_loss'] = float(trade_detail['stopLossOrder']['price'])

        elif "stopLossOrderID" in trade_detail:
            r = orders.OrderDetails(self.account_id, trade_detail['stopLossOrderID'])
            rv = self.oanda_api.request(r)
            trade_settings['stop_loss'] = float(rv['order']['price'])

        if "trailingStopLossOrder" in trade_detail:
            trade_settings['trailing_stop_distance'] = float(trade_detail['trailingStopLossOrder']['distance'])

        elif "trailingStopLossOrderID" in trade_detail:
            r = orders.OrderDetails(self.account_id, trade_detail['trailingStopLossOrderID'])
            rv = self.oanda_api.request(r)
            trade_settings['trailing_stop_distance'] = float(rv['order']['distance'])

        # print(json.dumps(trade_detail, indent=2))

        trade = BrokerPlatform.Trade(trade_detail['id'], trade_detail['instrument'], trade_detail['price'], trade_settings, self._update_trade_handler)

        if "takeProfitOrderID" in trade_detail:
            trade.take_profit_order_id = trade_detail['takeProfitOrderID']
        if "stopLossOrderID" in trade_detail:
            trade.stop_lost_order_id = trade_detail['stopLossOrderID']
        if "trailingStopLossOrderID" in trade_detail:
            trade.trailing_stop_order_id = trade_detail['trailingStopLossOrderID']

        trade.close_handler = self._trade_close_handler
        trade.modify_handler = None
        trade.open_price = float(trade_detail['price'])
        trade.price = float(trade_detail['price'])
        trade.unrealized_pl = float(trade_detail['unrealizedPL'])

        self.account.trades.append(trade)
        return trade

    def _convert_mid_candles_to_dataframe(self, candles):
        rows = []
        for candle in candles:
            row = {
                "timestamp": candle['time'],
                "open": candle['mid']['o'],
                "high": candle['mid']['h'],
                "low": candle['mid']['l'],
                "close": candle['mid']['c'],
                "volume": candle['volume'],
                "completed": candle['complete'],
            }
            rows.append(row)
        return pd.DataFrame(data=rows)

    def _remove_order_detail(self, order_id, reason=None):
        for order in self.account.orders:
            if order.order_id == order_id:
                order.canceled = True
                order.canceled_listener(order)
                self.account.orders.remove(order)
                break

    def _remove_trade_detail(self, trade_id):
        for trade in self.account.trades:
            if trade.trade_id == trade_id:
                trade.closed = True
                trade.closed_listener(trade)
                self.account.trades.remove(trade)
                break

    def _order_cancel_handler(self, order_id):
        r = orders.OrderCancel(self.account_id, order_id)
        rv = self.oanda_api.request(r)
        # if 'orderCancelTransaction' in rv:
        #     self._remove_order_detail(order_id)
        #     return
        if 'orderCancelRejectTransaction' in rv:
            print(json.dumps(rv['orderCancelRejectTransaction'], indent=2))
            raise Exception('Order cancel rejected by oanda!')

    def _trade_close_handler(self, trade_id):
        r = trades.TradeClose(self.account_id, trade_id)
        rv = self.oanda_api.request(r)
        # if 'orderFillTransaction' in rv:
        #     self._remove_trade_detail(trade_id)
        #     return
        if 'orderRejectTransaction' in rv:
            raise Exception('Trade close rejected by oanda!')

    def _trade_modify_handler(self, trade_id, trade_settings):
        pass

    def _update_instrument_handler(self, instrument):
        if 1000*(datetime.now().timestamp() - instrument.latest_update_datetime.timestamp()) < update_interval_threshold_ms:
            # print('skipped.', 1000*(datetime.now().timestamp() - instrument.latest_update_datetime.timestamp()))
            return
        params = {
            "granularity": "M1",
            "from": instrument.latest_candles_iso_time,
        }

        r = instruments.InstrumentsCandles(instrument.instrument_name, params)
        rv = self.oanda_api.request(r)
        candles = rv['candles']
        candles_df = self._convert_mid_candles_to_dataframe(candles)

        instrument.active_1m_candle = candles_df.loc[candles_df['completed'] == False]
        new_candles_df = candles_df.loc[candles_df['completed'] == True]
        instrument.recent_1m_candles = instrument.recent_1m_candles.append(new_candles_df)
        instrument.latest_candles_iso_time = candles[-1]['time']
        instrument.latest_update_datetime = datetime.now()

    def _update_account_handler(self):
        if 1000*(datetime.now().timestamp() - self.account.latest_update_datetime.timestamp()) < update_interval_threshold_ms:
            # print('skipped.', 1000*(datetime.now().timestamp() - self.account.latest_update_datetime.timestamp()))
            return
        r = accounts.AccountSummary(self.account_id)
        rv = self.oanda_api.request(r)
        account = rv['account']
        self.account.balance = account['balance']
        self.account.currency = account['currency']
        self.account.margin_rate = account['marginRate']
        self.account.margin_used = account['marginUsed']
        self.account.margin_available = account['marginAvailable']
        self.account.unrealized_pl = account['unrealizedPL']
        self.account.latest_update_datetime = datetime.now()

    def _update_trade_handler(self, trade):
        r = trades.TradeDetails(self.account_id, trade.trade_id)
        rv = self.oanda_api.request(r)

        trade_detail = rv['trade']
        # print(json.dumps(trade_detail, indent=2))

        trade_settings = {}
        trade_settings['units'] = float(trade_detail['currentUnits'])

        if "takeProfitOrder" in trade_detail:
            trade_settings['take_profit'] = float(trade_detail['takeProfitOrder']['price'])

        if "stopLossOrder" in trade_detail:
            trade_settings['stop_loss'] = float(trade_detail['stopLossOrder']['price'])

        if "trailingStopLossOrder" in trade_detail:
            trade_settings['trailing_stop_distance'] = float(trade_detail['trailingStopLossOrder']['distance'])

        trades.trade_settings = trade_settings
        trade.unrealized_pl = float(trade_detail['unrealizedPL'])

    def _loop(self):
        instruments_string = ','.join([i for i in self.instruments_watchlist])
        r = pricing.PricingInfo(self.account_id, params={"instruments":instruments_string})
        rv = self.oanda_api.request(r)
        # print(json.dumps(rv, indent=2))
        prices = rv['prices']
        # Update instruments
        for price in prices:
            instrument = self.instruments_watchlist[price['instrument']]
            instrument.current_closeout_bid = price['closeoutBid']
            instrument.current_closeout_ask = price['closeoutAsk']
            instrument.current_closeout_bid_ask_datetime = dateutil.parser.isoparse(price['time'])
            instrument.tradable = price['status'] == 'tradeable'
            # instrument.recent_1m_candles = self._convert_mid_candles_to_dataframe(candles)
            # print(instrument.recent_1m_candles)
            # print(instrument.recent_1m_candles.tail(1))

        # Update transactions
        r = transactions.TransactionsSinceID(self.account_id, {"id": self.latest_sync_transaction_id})
        rv = self.oanda_api.request(r)
        self.latest_sync_transaction_id = int(rv['lastTransactionID'])
        transaction_list = rv['transactions']
        # print(json.dumps(transaction_list, indent=2))
        for transaction in transaction_list:
            if transaction['type'] == 'ORDER_CANCEL' and transaction['reason'] != 'LINKED_TRADE_CLOSED' :
                # print(json.dumps(transaction, indent=2))
                self._remove_order_detail(transaction['orderID'], transaction['reason'])
            elif transaction['type'] == 'ORDER_FILL':
                # print(json.dumps(transaction, indent=2))
                full_price = transaction['fullPrice']
                closeout_bid = full_price['closeoutBid']
                closeout_ask = full_price['closeoutAsk']
                timestamp = dateutil.parser.isoparse(full_price['timestamp'])
                if 'tradesClosed' in transaction:
                    trades_closed = transaction['tradesClosed']
                    for trade_closed in trades_closed:
                        trade_id = trade_closed['tradeID']
                        for trade in self.account.trades:
                            # print(trade.trade_id)
                            # print(trade.trade_id == trade_id)
                            # print(trade.closed_listener)
                            if trade.trade_id == trade_id:
                                realized_pl = float(trade_closed['realizedPL'])
                                close_price = float(trade_closed['price'])
                                spread = float(trade_closed['halfSpreadCost'])
                                trade.closed = True
                                trade.closed_listener(trade, realized_pl, close_price, spread, timestamp)
                                self.account.trades.remove(trade)

                    # print(json.dumps(transaction['tradesClosed'], indent=2))
                if 'tradeOpened' in transaction:
                    trade_opened = transaction['tradeOpened']
                    for order in self.account.orders:
                        if order.order_id == transaction['orderID']:
                            trade_id = trade_opened['tradeID']
                            r = trades.TradeDetails(self.account_id, trade_id)
                            rv = self.oanda_api.request(r)
                            trade = self._import_trade_detail(rv['trade'])

                            # print(json.dumps(rv, indent=2))

                            order.filled_listener(order, trade)
                            self.account.orders.remove(order)
                    # print(json.dumps(transaction['tradeOpened'], indent=2))
