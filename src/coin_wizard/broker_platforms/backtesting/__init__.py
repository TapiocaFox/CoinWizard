#!/usr/bin/python3

import json, dateutil, pytz, random, time, traceback
import pandas as pd

from coin_wizard.historical_pair_data import get_historical_pair_data_pandas, plot_historical_pair_data
from coin_wizard.broker_platforms.backtesting.utils import translate_pair_to_splited, translate_pair_to_unsplited
import coin_wizard.broker_platform_objects as BrokerPlatform
from datetime import datetime, timedelta
from time import sleep
#
# update_interval_threshold_ms = 50
time_delta_15_seconds = timedelta(seconds=15)
time_delta_60_seconds = timedelta(seconds=60)
time_delta_7_days = timedelta(days=7)
utc = pytz.utc

half_spread_high_pip = 0.8
half_spread_low_pip = 0.6

class BrokerEventLoopAPI(BrokerPlatform.BrokerEventLoopAPI):
    hedging = False
    broker_settings_fields = ['balance', 'currency', 'margin_rate', 'start_year_utc', 'start_month_utc', 'start_day_utc', 'start_hour_utc', 'start_minute_utc', 'end_year_utc', 'end_month_utc', 'end_day_utc', 'end_hour_utc', 'end_minute_utc']
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms = 1000):
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
        self.account = BrokerPlatform.Account(self._update_account_handler)
        self.account.balance = float(broker_settings['balance'])
        self.account.currency = broker_settings['currency']
        self.account.margin_rate = broker_settings['margin_rate']
        self.account.margin_used = 0.0
        self.account.margin_available = float(broker_settings['balance'])
        self.account.unrealized_pl = 0.0
        self.ended_listener = None

    def order(self, instrument_name, order_settings, trade_settings):

        # price_bound = order_settings['bound']
        # units = trade_settings['units']
        # take_profit = trade_settings['take_profit']
        # stop_loss = trade_settings['stop_lost']
        # trailing_stop_distance = trade_settings['trailing_stop_distance']

        order = BrokerPlatform.Order('virtual_order_id', instrument_name, order_settings, trade_settings)
        self.account.orders.append(order)
        return order

    def getInstrument(self, instrument_name):
        if instrument_name in self.instruments_watchlist:
            return self.instruments_watchlist[instrument_name]

        instrument = BrokerPlatform.Instrument(instrument_name, self._update_instrument_handler)
        instrument.recent_1m_candles = get_historical_pair_data_pandas(translate_pair_to_unsplited(instrument_name), self.current_virtual_datetime - time_delta_7_days, self.current_virtual_datetime)
        latest_candle = instrument.recent_1m_candles.tail(1).iloc[0]

        open = latest_candle['open']
        high = latest_candle['high']
        low = latest_candle['low']
        close = latest_candle['close']

        half_spread = random.uniform(high*half_spread_low_pip, high*half_spread_high_pip)*0.0001
        price = 0.8*random.triangular(low, high)+0.2*random.triangular(min(open, close), max(open, close))
        # print(price, half_spread, price-half_spread, price+half_spread, open, high, low, close)
        instrument.current_closeout_bid = price-half_spread
        instrument.current_closeout_ask = price+half_spread
        instrument.current_closeout_bid_ask_datetime = self.current_virtual_datetime
        # raise
        self.instruments_watchlist[instrument_name] = instrument
        return instrument

    # For Neural Net
    def resetByDate(self, start_datetime, end_datetime):
        pass

    def onEnded(self, ended_listener):
        self.ended_listener = ended_listener

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
        for instrument_name in self.instruments_watchlist:
            instrument = self.instruments_watchlist[instrument_name]
            latest_candle_df = get_historical_pair_data_pandas(translate_pair_to_unsplited(instrument_name), self.current_virtual_datetime - time_delta_60_seconds, self.current_virtual_datetime).tail(1)
            # print(latest_candle_df)
            # print(self.current_virtual_datetime - time_delta_60_seconds)
            # print(self.current_virtual_datetime)
            # print(instrument.recent_1m_candles)
            latest_candle = latest_candle_df.iloc[0]
            # print(latest_candle)
            # print(latest_candle['timestamp'] != instrument.recent_1m_candles.tail(1).iloc[0]['timestamp'])
            if latest_candle['timestamp'] != instrument.recent_1m_candles.tail(1).iloc[0]['timestamp']:
                instrument.recent_1m_candles.loc[len(instrument.recent_1m_candles)] = latest_candle
        self.current_virtual_datetime += time_delta_15_seconds

    def _loop_wrapper(self):
        self.before_loop()
        self._loop()
        print(1000*(self.current_virtual_datetime.timestamp() - self.latest_every_15_second_loop_datetime.timestamp()))
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
