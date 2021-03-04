#!/usr/bin/python3

import json, dateutil, pytz
import pandas as pd

from coin_wizard.historical_pair_data import get_historical_pair_data_pandas, plot_historical_pair_data
from coin_wizard.broker_platforms.backtesting.utils import translate_pair_to_splited, translate_pair_to_unsplited
import coin_wizard.broker_platform_objects as BrokerPlatform
from datetime import datetime, timedelta
from time import sleep
#
# update_interval_threshold_ms = 50
time_delta_15_seconds = timedelta(seconds=15)
time_delta_7_days = timedelta(days=7)
utc = pytz.utc

class BrokerEventLoopAPI(BrokerPlatform.BrokerEventLoopAPI):
    hedging = False
    broker_settings_fields = ['balance', 'currency', 'margin_rate', 'start_year_utc', 'start_month_utc', 'start_day_utc', 'start_hour_utc', 'start_minute_utc', 'end_year_utc', 'end_month_utc', 'end_day_utc', 'end_hour_utc', 'end_minute_utc']
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms = 0):
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
        self.instruments_watchlist[instrument_name] = instrument
        return instrument

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

    def _loop_wrapper(self):
        self.before_loop()
        self._loop()

        # Fire every_15_second_listener if needed.
        if 1000*(self.current_virtual_datetime.timestamp() - self.latest_every_15_second_loop_datetime.timestamp()) > 15000:
            self.every_15_second_listener(self)
            self.latest_every_15_second_loop_datetime = self.current_virtual_datetime
        self.loop_listener(self)
        self.after_loop()
        end_loop_timestamp = datetime.now().timestamp()
        time_passed_ms = (end_loop_timestamp - self.latest_loop_datetime.timestamp())*1000
        # print('time_passed_ms', time_passed_ms, self.loop_interval_ms)
        if(time_passed_ms < self.loop_interval_ms):
            time.sleep(0.001*(self.loop_interval_ms - time_passed_ms))

    def _loop(self):
        pass

    # For Neural Net
    def resetByDate(self, start_datetime, end_datetime):
        pass
