#!/usr/bin/python3

class TradingAgent(object):
    def __init__(self, agent_directory):
        print('Started directory:', agent_directory)
        self.every_15_second_loop_count = 1
        self.buy_or_sell = 'SELL'

    def _order_canceled_listener(self, order, reason):
        pass

    def _order_filled_listener(self, order, trade):
        if trade != None:
            trade.onReduced(self._trade_reduced_listener)
            trade.onClosed(self._trade_closed_listener)

    def _trade_reduced_listener(self, trade, units, realized_pl, close_price, spread, timestamp):
        pass

    def _trade_closed_listener(self, trade, realized_pl, close_price, spread, timestamp):
        pass

    def _every_15_second_loop(self, BrokerAPI):
        # 4*15 = 60 second = 1 min
        if self.every_15_second_loop_count%4 == 0 :
            print(self.every_15_second_loop_count*15, 'secs passed.')
            if self.buy_or_sell == 'SELL':
                # next 1 min
                self.buy_or_sell = 'BUY'
                # Sell 2 units
                order = BrokerAPI.order('EUR_USD', {"type": "market"}, {"units": -2})
                # Register event listener on .... then call such listener
                order.onCanceled(self._order_canceled_listener)
                order.onFilled(self._order_filled_listener)
            else:
                # next 1 min
                self.buy_or_sell = 'SELL'
                # Buy 2 units
                order = BrokerAPI.order('EUR_USD', {"type": "market"}, {"units": 2})
                order.onCanceled(self._order_canceled_listener)
                order.onFilled(self._order_filled_listener)

        self.every_15_second_loop_count += 1

    # Entry point here
    def run(self, BrokerAPI):
        # Register event listener on .... then call such listener "_every_15_second_loop" self mean the class itself.
        BrokerAPI.onEvery15Second(self._every_15_second_loop)

    def stop_running(self, BrokerAPI):
        # Close all trades and orders
        account = BrokerAPI.getAccount()
        orders = account.getOrders()
        trades = account.getTrades()
        for order in orders:
            # print(order.getInstrumentName(), order.getOrderSettings(), order.getTradeSettings())
            order.onCanceled(self._order_canceled_listener)
            order.onFilled(self._order_filled_listener)
            order.cancel()

        for trade in trades:
            # print(trade.getInstrumentName(), trade.getTradeSettings())
            trade.onReduced(self._trade_reduced_listener)
            trade.onClosed(self._trade_closed_listener)
            trade.close()

    def train(self, BrokerAPI):
        pass

    def stop_training(self, BrokerAPI):
        pass

    def test(self, BacktestBrokerAPI):
        self.run(BacktestBrokerAPI)

    def stop_testing(self, BacktestBrokerAPI):
        self.stop_running(BacktestBrokerAPI)
