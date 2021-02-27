#!/usr/bin/python3

import coin_wizard.broker_platform_object as BrokerPlatform
from time import sleep
from coin_wizard.broker_platforms.backtest.supported_pairs import SupportedPairs

class BrokerEventLoopAPI(BrokerPlatform.BrokerEventLoopAPI):
    SupportedPairs = SupportedPairs
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms= 50):
        super().__init__(self, before_loop, after_loop, broker_settings, loop_interval_ms)

    def __loop(self):
        pass
