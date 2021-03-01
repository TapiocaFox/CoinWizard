#!/usr/bin/python3

import coin_wizard.broker_platform_objects as BrokerPlatform
from time import sleep

timedelta = None

class BrokerEventLoopAPI(BrokerPlatform.BrokerEventLoopAPI):
    hedging = True
    broker_settings_fields = ['balance', 'currency', 'margin_rate', 'start_year', 'start_month', 'start_day', 'start_hour', 'start_minute', 'end_year', 'end_month', 'end_day', 'end_hour', 'end_minute']
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms = 5):
        super().__init__(before_loop, after_loop, broker_settings, loop_interval_ms)
        self.balance = float(broker_settings['balance'])
        self.currency = broker_settings['currency'].lower()
        self.margin_rate = float(broker_settings['margin_rate'])

        self.start_datetime = None
        self.end_datetime = None
        self.current_datetime = None

    def order(self, order_settings, trade_settings):
        pass

    def getInstrument(self, instrument_name):
        pass

    def getAccount(self):
        pass

    def _loop(self):
        pass

    # For Neural Net
    def resetByDate(self, start_datetime, end_datetime):
        pass
