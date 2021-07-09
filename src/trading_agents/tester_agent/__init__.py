#!/usr/bin/python3

import time
import plotly.graph_objects as go
import pytz
eastern = pytz.timezone('US/Eastern')
utc = pytz.utc

from datetime import datetime
from coin_wizard.historical_pair_data import plot_historical_pair_data
# import state manager
t = None

c = 0
class TradingAgent(object):
    def __init__(self, agent_directory):
        print('Started directory:', agent_directory)

    def _order_canceled_listener(self, order, reason):
        pass

    def _order_filled_listener(self, order, trade):
        # print('')
        # print('An order filled.')
        # print(' Order settings:', order.order_settings)
        # print(' Order trade settings:', order.trade_settings)
        # # global t
        if trade != None:
        #     print(' Instrument:', trade.instrument_name)
        #     print(' Open price:', trade.getOpenPrice())
        #     print(' Unrealized PL:', trade.getUnrealizedPL())
        #     print(' Trade settings:', trade.getTradeSettings())
        #     print(' Account balance:', self.account.getBalance())
        #     print(' Account unrealized PL:',self.account.getUnrealizedPL())
        #     print(' Account margin available:', self.account.getMarginAvailable())
        #     print(' Account margin used:', self.account.getMarginUsed())
            trade.onReduced(self._trade_reduced_listener)
            trade.onClosed(self._trade_closed_listener)


    def _trade_reduced_listener(self, trade, units, realized_pl, close_price, spread, timestamp):
        pass

        # print('')
        # print('A trade reduced.')
        # print(' Instrument:', trade.instrument_name)
        # print(' Units:', units)
        # print(' Open price:', trade.getOpenPrice())
        # print(' Close price:', close_price)
        # print(' Realized PL:', realized_pl)
        # print(' Trade settings:', trade.getTradeSettings())
        # print(' Account balance:', self.account.getBalance())
        # print(' Account unrealized PL:',self.account.getUnrealizedPL())
        # print(' Account margin available:', self.account.getMarginAvailable())
        # print(' Account margin used:', self.account.getMarginUsed())

    def _trade_closed_listener(self, trade, realized_pl, close_price, spread, timestamp):
        pass

        # print('')
        # print('A trade closed.')
        # print(' Instrument:', trade.instrument_name)
        # print(' Open price:', trade.getOpenPrice())
        # print(' Close price:', close_price)
        # print(' Realized PL:', realized_pl)
        # print(' Trade settings:', trade.getTradeSettings())
        # print(' Account balance:', self.account.getBalance())
        # print(' Account unrealized PL:',self.account.getUnrealizedPL())
        # print(' Account margin available:', self.account.getMarginAvailable())
        # print(' Account margin used:', self.account.getMarginUsed())


        # print(datetime.now().timestamp()-timestamp.timestamp())

    def _run_loop(self, BrokerAPI):
        instrument = BrokerAPI.getInstrument('EUR_USD')
        print(instrument.getRecentCandles(10, granularity='M15'))
        print(instrument.getRecentCandles(10, granularity='M5'))
        print(instrument.getRecent1MCandles(10))
        print(instrument.getActive1MCandle())
        print(instrument.getCurrentCloseoutBidAsk())
        # print(instrument.isTradable())
        print(self.account.orders)
        print(self.account.trades)
        pass

    def _every_15_second_loop(self, BrokerAPI):
        global c
        if c < 6:
            # print(c)
            c += 1
            return
        print('15 second passed.', datetime.now())
        orders = self.account.getOrders()
        trades = self.account.getTrades()
        for order in orders:
            print(order.getInstrumentName(), order.getOrderSettings(), order.getTradeSettings())
            order.onCanceled(self._order_canceled_listener)
            order.onFilled(self._order_filled_listener)
            order.cancel()

        for trade in trades:
            print(trade.getInstrumentName(), trade.getTradeSettings())
            trade.onReduced(self._trade_reduced_listener)
            trade.onClosed(self._trade_closed_listener)
            trade.close()

    def run(self, BrokerAPI):
        # BrokerAPI.resetByDatetime(eastern.localize(datetime(2020, 1, 8, 0, 0)), eastern.localize(datetime(2020, 1, 11, 23, 59)))
        account = BrokerAPI.getAccount()
        self.account = account
        orders = account.getOrders()
        trades = account.getTrades()
        for order in orders:
            print(order.getInstrumentName(), order.getOrderSettings(), order.getTradeSettings())
            order.onCanceled(self._order_canceled_listener)
            order.onFilled(self._order_filled_listener)
            order.cancel()

        for trade in trades:
            print(trade.getInstrumentName(), trade.getTradeSettings())
            trade.onReduced(self._trade_reduced_listener)
            trade.onClosed(self._trade_closed_listener)
            trade.close()

        orders = account.getOrders()
        trades = account.getTrades()

        print(orders, trades)

        order = BrokerAPI.order('EUR_USD', {"type": "stop", "price": 2, "bound": 2.1}, {"units": 1, "take_profit": 2, "stop_lost": 0.5, "trailing_stop_distance": 0.001})
        order.onCanceled(self._order_canceled_listener)
        order.onFilled(self._order_filled_listener)
        print(order.order_id)

        order = BrokerAPI.order('EUR_USD', {"type": "market"}, {"units": 4, "take_profit": 2, "stop_lost": 0.5, "trailing_stop_distance": 0.1})
        order.onCanceled(self._order_canceled_listener)
        order.onFilled(self._order_filled_listener)
        print(order.order_id)

        order = BrokerAPI.order('EUR_USD', {"type": "market"}, {"units": -1, "take_profit": 0.543, "stop_lost": 2, "trailing_stop_distance": 0.1})
        order.onCanceled(self._order_canceled_listener)
        order.onFilled(self._order_filled_listener)
        print(order.order_id)


        order = BrokerAPI.order('EUR_USD', {"type": "market"}, {"units": -8000, "take_profit": 0.543, "stop_lost": 2, "trailing_stop_distance": 0.1})
        order.onCanceled(self._order_canceled_listener)
        order.onFilled(self._order_filled_listener)
        print(order.order_id)

        order = BrokerAPI.order('EUR_USD', {"type": "market"}, {"units": 1, "take_profit": 2, "stop_lost": 0.5, "trailing_stop_distance": 0.1})
        order.onCanceled(self._order_canceled_listener)
        order.onFilled(self._order_filled_listener)
        print(order.order_id)

        # order = BrokerAPI.order('EUR_USD', {"type": "market"}, {"units": -2})
        # print(order.order_id)

        print(order.getOrderSettings())
        print(order.getTradeSettings())

        instrument = BrokerAPI.getInstrument('EUR_USD')
        BrokerAPI.onLoop(self._run_loop)
        BrokerAPI.onEvery15Second(self._every_15_second_loop)
        # BrokerAPI.resetByDatetime(eastern.localize(datetime(2020, 1, 8, 0, 0)), eastern.localize(datetime(2020, 1, 11, 23, 59)))
        # self.account = BrokerAPI.getAccount()

    def stop_running(self, BrokerAPI):
        print('Agent stopped.')

    def train(self, BrokerAPI):
        plot_historical_pair_data('eurusd', eastern.localize(datetime(2021, 1, 8, 0, 0)), eastern.localize(datetime(2021, 1, 11, 23, 59)), 'US/Eastern')

    def stop_training(self, BrokerAPI):
        pass

    def test(self, BacktestBrokerAPI):
        self.run(BacktestBrokerAPI)

    def stop_testing(self, BacktestBrokerAPI):
        self.stop_running(BacktestBrokerAPI)
