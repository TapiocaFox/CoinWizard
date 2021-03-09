
import coin_wizard.broker_platforms.backtesting as backtesting

class BrokerEventLoopAPI(backtesting.BrokerEventLoopAPI):
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms = 1000, hedging=True):
        super().__init__(before_loop, after_loop, broker_settings, loop_interval_ms, hedging)
