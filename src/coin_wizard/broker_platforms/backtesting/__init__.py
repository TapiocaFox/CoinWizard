#!/usr/bin/python3

import json, dateutil, pytz, random, time, traceback
import pandas as pd

from coin_wizard.historical_pair_data import get_historical_pair_data_pandas, plot_historical_pair_data
from coin_wizard.utils import translate_pair_to_splited, translate_pair_to_unsplited
import coin_wizard.broker_platform_objects as BrokerPlatform
from datetime import datetime, timedelta
from time import sleep

price_list = []
#
# update_interval_threshold_ms = 50
time_delta_15_seconds = timedelta(seconds=15)
time_delta_60_seconds = timedelta(seconds=60)
time_delta_7_days = timedelta(days=7)
utc = pytz.utc

half_spread_high_pip = 0.8
half_spread_low_pip = 0.6

min_trailing_stop_distance = 0.0005

class BrokerEventLoopAPI(BrokerPlatform.BrokerEventLoopAPI):
    hedging = False
    broker_settings_fields = ['balance', 'currency', 'margin_rate', 'start_year_utc', 'start_month_utc', 'start_day_utc', 'start_hour_utc', 'start_minute_utc', 'end_year_utc', 'end_month_utc', 'end_day_utc', 'end_hour_utc', 'end_minute_utc']
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms = 1000, hedging=True):
        super().__init__(before_loop, after_loop, broker_settings, loop_interval_ms)
        self.instruments_watchlist = {}
        self.current_virtual_datetime = utc.localize(datetime(
                                            int(broker_settings['start_year_utc']),
                                            int(broker_settings['start_month_utc']),
                                            int(broker_settings['start_day_utc']),
                                            int(broker_settings['start_hour_utc']),
                                            int(broker_settings['start_minute_utc'])
                                        ))

        self.end_datetime = utc.localize(datetime(
                                            int(broker_settings['end_year_utc']),
                                            int(broker_settings['end_month_utc']),
                                            int(broker_settings['end_day_utc']),
                                            int(broker_settings['end_hour_utc']),
                                            int(broker_settings['end_minute_utc'])
                                        ))
        self.hedging = hedging
        self.account = BrokerPlatform.Account(self._update_account_handler)
        self.account.balance = float(broker_settings['balance'])
        self.account.currency = broker_settings['currency']
        self.account.margin_rate = float(broker_settings['margin_rate'])
        self.account.margin_used = 0.0
        self.account.margin_available = float(broker_settings['balance'])
        self.account.unrealized_pl = 0.0
        self.ended_listener = None

    def order(self, instrument_name, order_settings, trade_settings):
        instrument = self.getInstrument(instrument_name)
        bid, ask, d = instrument.getCurrentCloseoutBidAsk()

        # Start validations
        units = trade_settings['units']

        if "take_profit" in trade_settings:
            take_profit = trade_settings['take_profit']
        else:
            take_profit = None

        if "stop_lost" in trade_settings:
            stop_loss = trade_settings['stop_lost']
        else:
            stop_loss = None

        if "trailing_stop_distance" in trade_settings:
            trailing_stop_distance = trade_settings['trailing_stop_distance']
        else:
            trailing_stop_distance = None

        if order_settings['type'] == 'market':
            pass

        elif order_settings['type'] == 'stop':
            price = order_settings['price']
            if 'bound' in order_settings:
                price_bound = order_settings['bound']
            else:
                price_bound = None
            if units > 0:
                if price < ask:
                    raise Exception('Price('+str(price)+') cannot smaller then ask price('+str(ask)+').')
                if price_bound!=None and price_bound < price:
                    raise Exception('Price bound('+str(price_bound)+') cannot smaller then price('+str(price)+').')
            elif units < 0:
                if price > bid:
                    raise Exception('Price('+str(price)+') cannot greater then bid price('+str(bid)+').')
                if price_bound!=None and price_bound > price:
                    raise Exception('Price bound('+str(price_bound)+') cannot greater then price('+str(price)+').')
        else:
            raise Exception('Does not supoort type "'+ order_settings['type']+'".')

        # Validating trade setting
        if int(units) - units != 0:
            raise Exception('Order units must be integer.')
        elif units == 0:
            raise Exception('Order units cannot be zero.')

        elif units > 0:
            if take_profit!=None and take_profit <= ask:
                raise Exception('Take profit('+str(take_profit)+') cannot smaller then ask price('+str(ask)+').')
            if stop_loss!=None and stop_loss >= bid:
                raise Exception('Stop loss('+str(stop_loss)+') cannot greater then bid price('+str(bid)+').')
            if trailing_stop_distance!=None and trailing_stop_distance < min_trailing_stop_distance:
                raise Exception('Trailing stop distance('+str(trailing_stop_distance)+') cannot smaller then mininum trailing stop distance('+str(min_trailing_stop_distance)+').')
        elif units < 0:
            if take_profit!=None and take_profit >= bid:
                raise Exception('Take profit('+str(take_profit)+') cannot greater then bid price('+str(bid)+').')
            if stop_loss!=None and stop_loss <= ask:
                raise Exception('Stop loss('+str(stop_loss)+') cannot smaller then ask price('+str(ask)+').')
            if trailing_stop_distance!=None and trailing_stop_distance < min_trailing_stop_distance:
                raise Exception('Trailing stop distance('+str(trailing_stop_distance)+') cannot smaller then mininum trailing stop distance('+str(min_trailing_stop_distance)+').')

        if trailing_stop_distance != None and trailing_stop_distance <= 0:
            raise Exception('Trailing stop distance('+str(trailing_stop_distance)+') should greater then zero.')

        order = BrokerPlatform.Order('virtual_order_id', instrument_name, order_settings, trade_settings)
        self.account.orders.append(order)
        return order

    def getInstrument(self, instrument_name):
        if instrument_name in self.instruments_watchlist:
            return self.instruments_watchlist[instrument_name]

        instrument = BrokerPlatform.Instrument(instrument_name, self._update_instrument_handler)
        instrument.recent_1m_candles = get_historical_pair_data_pandas(translate_pair_to_unsplited(instrument_name), self.current_virtual_datetime - time_delta_7_days, self.current_virtual_datetime)
        instrument.future_1m_candles = get_historical_pair_data_pandas(translate_pair_to_unsplited(instrument_name), self.current_virtual_datetime , self.end_datetime - time_delta_7_days)
        latest_candle = instrument.recent_1m_candles.tail(1).iloc[0]

        open = latest_candle['open']
        high = latest_candle['high']
        low = latest_candle['low']
        close = latest_candle['close']

        half_spread = random.uniform(high*half_spread_low_pip, high*half_spread_high_pip)*0.0001
        price = 0.8*random.triangular(low, high)+0.2*random.triangular(min(open, close), max(open, close))
        # print(price, half_spread, price-half_spread, price+half_spread, open, high, low, close)
        instrument.current_closeout_bid = round(float(price-half_spread), 6)
        instrument.current_closeout_ask = round(float(price+half_spread), 6)
        instrument.current_closeout_bid_ask_datetime = self.current_virtual_datetime
        instrument.tradable = True
        instrument.quaters = 0
        # raise
        self.instruments_watchlist[instrument_name] = instrument
        return instrument

    # For Neural Net
    def resetByDate(self, start_datetime, end_datetime):
        pass

    def onEnded(self, ended_listener):
        self.ended_listener = ended_listener

    def _order_cancel_handler(self, order):
        order.canceled = True
        self.account.orders.remove(order)
        order.canceled_listener(order, 'Canceled by user.')

    def _trade_close_handler(self, trade):
        instrument = self.instruments_watchlist[trade.instrument_name]
        units = trade.trade_settings['units']
        ask_price = instrument.current_closeout_ask
        bid_price = instrument.current_closeout_bid
        spread = ask_price-bid_price
        unrealized_pl = 0.0
        if units > 0:
            unrealized_pl = units*(bid_price - trade.open_price)
            trade.unrealized_pl = unrealized_pl
            trade.closed = True
            self.account.trades.remove(trade)
            self.account.balance += unrealized_pl
            self.account.unrealized_pl -= unrealized_pl
            trade.closed_listener(trade, unrealized_pl, bid_price, spread, self.current_virtual_datetime)
        else:
            unrealized_pl = -units*(trade.open_price - ask_price)
            trade.unrealized_pl = unrealized_pl
            trade.closed = True
            self.account.trades.remove(trade)
            self.account.balance += unrealized_pl
            self.account.unrealized_pl -= unrealized_pl
            trade.closed_listener(trade, unrealized_pl, ask_price, spread, self.current_virtual_datetime)

    def _trade_reduce_handler(self, trade, units):
        pass

    def _trade_modify_handler(self, trade, trade_settings):
        pass

    def _update_instrument_handler(self, instrument):
        pass # No need for backtesting.

    def _update_account_handler(self):
        pass # No need for backtesting.

    def _update_trade_handler(self, trade):
        pass # No need for backtesting.

    # def _trade_close(self, trade):
    #     trade.closed = True
    #     self.account.trades.remove(trade)
    #     self.account.balance += unrealized_pl
    #     trade.closed_listener(trade, unrealized_pl, bid_price, spread, self.current_virtual_datetime)

    def _loop(self):
        # Update instruments
        for instrument_name in self.instruments_watchlist:
            instrument = self.instruments_watchlist[instrument_name]
            latest_candle_df = instrument.future_1m_candles
            # latest_candle_df = get_historical_pair_data_pandas(translate_pair_to_unsplited(instrument_name), self.current_virtual_datetime - time_delta_60_seconds, self.current_virtual_datetime).tail(1)
            latest_candle_df = latest_candle_df[latest_candle_df['timestamp'] <= self.current_virtual_datetime]
            latest_candle_df = latest_candle_df[latest_candle_df['timestamp'] >= self.current_virtual_datetime - time_delta_60_seconds]
            # print(latest_candle_df)
            # print(self.current_virtual_datetime - time_delta_60_seconds)
            # print(self.current_virtual_datetime)
            # print(instrument.recent_1m_candles)
            instrument.active_1m_candle = latest_candle_df.tail(1).reset_index(drop=True)
            latest_candle = latest_candle_df.tail(1)
            # print(self.current_virtual_datetime)
            if latest_candle.empty:
                continue

            latest_candle = latest_candle.iloc[0]
            # print(self.current_virtual_datetime.timestamp() - latest_candle['timestamp'].to_pydatetime().timestamp())

            if self.current_virtual_datetime.timestamp() - latest_candle['timestamp'].to_pydatetime().timestamp() >= 60:
                if instrument.tradable:
                    instrument.tradable = False
                    instrument.recent_1m_candles.loc[len(instrument.recent_1m_candles)] = latest_candle_df.head(1).iloc[0]
            else:
                instrument.tradable = True
                quaters_of_1m = instrument.quaters
                open = latest_candle['open']
                high = latest_candle['high']
                low = latest_candle['low']
                close = latest_candle['close']
                half_spread = random.uniform(high*half_spread_low_pip, high*half_spread_high_pip)*0.0001
                bound_to_close = open+0.25*(close-open)*quaters_of_1m
                # print(quaters_of_1m)
                price = 0.7*random.triangular(low, high)+0.2*random.triangular(min(bound_to_close, close), max(bound_to_close, close))+0.1*close
                instrument.current_closeout_bid = round(float(price-half_spread), 6)
                instrument.current_closeout_ask = round(float(price+half_spread), 6)
                instrument.current_closeout_bid_ask_datetime = self.current_virtual_datetime
                instrument.quaters += 1
                # print(latest_candle)
                # print(latest_candle['timestamp'] != instrument.recent_1m_candles.tail(1).iloc[0]['timestamp'])
                if len(latest_candle_df) >= 2:
                    instrument.quaters = 0
                    instrument.recent_1m_candles.loc[len(instrument.recent_1m_candles)] = latest_candle_df.head(1).iloc[0]
        self.current_virtual_datetime += time_delta_15_seconds

        # Update trades, caluculate PL.
        unrealized_pl_sum = 0.0
        margin_used_sum = 0.0

        for trade in self.account.trades:
            instrument = self.instruments_watchlist[trade.instrument_name]
            units = trade.trade_settings['units']
            ask_price = instrument.current_closeout_ask
            bid_price = instrument.current_closeout_bid
            spread = ask_price-bid_price
            unrealized_pl = 0.0

            # Check if we need to close trade
            trade_settings = trade.trade_settings
            if units > 0:
                unrealized_pl = units*(bid_price - trade.open_price)
                trade.unrealized_pl = unrealized_pl
                if "take_profit" in trade_settings:
                    take_profit = trade_settings['take_profit']
                    if bid_price >= take_profit:
                        trade.closed = True
                        self.account.trades.remove(trade)
                        self.account.balance += unrealized_pl
                        trade.closed_listener(trade, unrealized_pl, bid_price, spread, self.current_virtual_datetime)
                        continue
                if "stop_lost" in trade_settings:
                    stop_loss = trade_settings['stop_lost']
                    if bid_price <= stop_loss:
                        trade.closed = True
                        self.account.trades.remove(trade)
                        self.account.balance += unrealized_pl
                        trade.closed_listener(trade, unrealized_pl, bid_price, spread, self.current_virtual_datetime)
                        continue
                if "trailing_stop_distance" in trade_settings:
                    trailing_stop_distance = trade_settings['trailing_stop_distance']
                    trade.trailing_stop_value = max(trade.trailing_stop_value, bid_price-trailing_stop_distance)
                    trailing_stop_value = trade.trailing_stop_value
                    if bid_price <= trailing_stop_value:
                        trade.closed = True
                        self.account.trades.remove(trade)
                        self.account.balance += unrealized_pl
                        trade.closed_listener(trade, unrealized_pl, bid_price, spread, self.current_virtual_datetime)
                        continue
            else:
                unrealized_pl = -units*(trade.open_price - ask_price)
                trade.unrealized_pl = unrealized_pl
                if "take_profit" in trade_settings:
                    take_profit = trade_settings['take_profit']
                    if ask_price <= take_profit:
                        trade.closed = True
                        self.account.trades.remove(trade)
                        self.account.balance += unrealized_pl
                        trade.closed_listener(trade, unrealized_pl, ask_price, spread, self.current_virtual_datetime)
                        continue
                if "stop_lost" in trade_settings:
                    stop_loss = trade_settings['stop_lost']
                    if ask_price >= stop_loss:
                        trade.closed = True
                        self.account.trades.remove(trade)
                        self.account.balance += unrealized_pl
                        trade.closed_listener(trade, unrealized_pl, ask_price, spread, self.current_virtual_datetime)
                        continue
                if "trailing_stop_distance" in trade_settings:
                    trailing_stop_distance = trade_settings['trailing_stop_distance']
                    trade.trailing_stop_value = max(trade.trailing_stop_value, bid_price-trailing_stop_distance)
                    trailing_stop_value = trade.trailing_stop_value
                    if ask_price >= trailing_stop_value:
                        trade.closed = True
                        self.account.trades.remove(trade)
                        self.account.balance += unrealized_pl
                        trade.closed_listener(trade, unrealized_pl, ask_price, spread, self.current_virtual_datetime)
                        continue
            unrealized_pl_sum += unrealized_pl
            margin_used_sum += units*(ask_price+bid_price)/2.0

        self.account.unrealized_pl = unrealized_pl_sum
        self.account.margin_used = margin_used_sum
        # print(unrealized_pl_sum, margin_used_sum)
        # Check account states. Margin Closeout etc.
        equity = self.account.balance+margin_used_sum

        if margin_used_sum* 0.5 > equity:
            pass

        # Update orders.
        orders_to_be_canceled = []
        orders_to_be_filled = []
        for order in self.account.orders:
            # print(order)
            instrument_name = order.instrument_name
            instrument = self.instruments_watchlist[instrument_name]

            order_settings = order.order_settings
            trade_settings = order.trade_settings

            order_type = order_settings['type']
            units = order.trade_settings['units']
            ask_price = instrument.current_closeout_ask
            bid_price = instrument.current_closeout_bid

            trade_settings['current_units'] = units

            if order_type == 'market':
                if units > 0:
                    unrealized_pl = units*(bid_price - ask_price)
                    trade = BrokerPlatform.Trade('virtual_trade_id', instrument_name, ask_price, trade_settings, self._update_trade_handler)
                    trade.open_price = ask_price
                    trade.margin_rate = self.account.margin_rate
                    trade.unrealized_pl = unrealized_pl
                    trade.reduce_handler = self._trade_reduce_handler
                    trade.close_handler = self._trade_close_handler

                    if 'trailing_stop_distance' in trade.trade_settings:
                        trade.trailing_stop_value = bid_price - trade.trade_settings['trailing_stop_distance']

                    if 'stop_loss' in trade.trade_settings:
                        if trade.trade_settings['stop_loss'] >= bid_price:
                            orders_to_be_canceled.append([order, 'Long stop loss initial value must smaller than current bid price.'])
                            continue

                    if 'take_profit' in trade.trade_settings:
                        if trade.trade_settings['take_profit'] <= ask_price:
                            orders_to_be_canceled.append([order, 'Long take profit initial value must greater than current ask price.'])
                            continue

                    orders_to_be_filled.append([order, trade])
                else:
                    unrealized_pl = units*(ask_price - bid_price)
                    trade = BrokerPlatform.Trade('virtual_trade_id', instrument_name, bid_price, trade_settings, self._update_trade_handler)
                    trade.bid_price = ask_price
                    trade.margin_rate = self.account.margin_rate
                    trade.unrealized_pl = unrealized_pl
                    trade.reduce_handler = self._trade_reduce_handler
                    trade.close_handler = self._trade_close_handler

                    if 'trailing_stop_distance' in trade.trade_settings:
                        trade.trailing_stop_value = ask_price + trade.trade_settings['trailing_stop_distance']

                    if 'stop_loss' in trade.trade_settings:
                        if trade.trade_settings['stop_loss'] <= ask_price:
                            orders_to_be_canceled.append([order, 'Short stop loss initial value must greater than current ask price.'])
                            continue

                    if 'take_profit' in trade.trade_settings:
                        if trade.trade_settings['take_profit'] >= bid_price:
                            orders_to_be_canceled.append([order, 'Short take profit initial value must smaller than current bid price.'])
                            continue

                    orders_to_be_filled.append([order, trade])

            elif order_type == 'stop':
                price = order_settings['price']
                if 'bound' in order_settings:
                    price_bound = order_settings['bound']
                else:
                    pass
        # Fill orders and cancel orders.
        for order_to_be_filled in orders_to_be_filled:
            order = order_to_be_filled[0]
            trade = order_to_be_filled[1]
            units = trade.trade_settings['units']
            open_price = trade.open_price
            margin_rate = trade.margin_rate

            if units*open_price*margin_rate > self.account.margin_available:
                order_to_be_canceled.append([order, 'Not enough Margin available.'])
            else:
                self.account.unrealized_pl += trade.unrealized_pl
                self.account.trades.append(trade)
                self.account.orders.remove(order)
                order.filled = True
                order.filled_listener(order, trade)

        for order_to_be_canceled in orders_to_be_canceled:
            order = order_to_be_filled[0]
            reason = order_to_be_filled[1]
            order.canceled = True
            self.account.orders.remove(order)
            order.canceled_listener(order, reason)

    def _loop_wrapper(self):
        self.before_loop()
        self._loop()
        # print(1000*(self.current_virtual_datetime.timestamp() - self.latest_every_15_second_loop_datetime.timestamp()))
        # Fire every_15_second_listener if needed.
        if 1000*(self.current_virtual_datetime.timestamp() - self.latest_every_15_second_loop_datetime.timestamp()) >= 14999:
            self.every_15_second_listener(self)
            self.latest_every_15_second_loop_datetime = self.current_virtual_datetime
        self.loop_listener(self)
        self.after_loop()
        end_loop_timestamp = datetime.now().timestamp()
        time_passed_ms = (end_loop_timestamp - self.latest_loop_datetime.timestamp())*1000
        # print('time_passed_ms', time_passed_ms, self.loop_interval_ms)
        if(time_passed_ms < self.loop_interval_ms):
            time.sleep(0.001*(self.loop_interval_ms - time_passed_ms))

    def _run_loop(self):
        self.stopped = False
        self.latest_loop_datetime = datetime.now()
        self.latest_every_15_second_loop_datetime = self.current_virtual_datetime
        loop_failed_count = 0
        while True:
            if self.stopped:
                return
            try:
                self._loop_wrapper()
                loop_failed_count = 0
            except Exception as err:
                loop_failed_count += 1
                traceback.print_tb(err.__traceback__)
                print(err)
                print('A loop skipped with a exception. This is a '+str(loop_failed_count)+' times failure.')
                if loop_failed_count > 3:
                    print('Too many failures, skipped next loop.')
                    break
            self.latest_loop_datetime = datetime.now()
