#!/usr/bin/python3

class BrokerEventLoopAPI(object):
    def __init__(self, before_loop, after_loop, broker_settings, loop_interval_ms= 50):
        self.before_loop = before_loop
        self.after_loop = after_loop
        self.latest_loop_timestamp = None
        self.loop_listener = None
        self.loop_interval_ms = loop_interval_ms

    def order(self, order_settings, order_callback):
        pass

    def getPairPrice(self, pair_name):
        pass

    def onLoop(self, loop_listener):
        self.loop_listener = loop_listener

    def __stop(self):
        pass

    def __loop(self):
        pass

    def __loop_wrapper(self):
        self.before_loop()
        # Do something
        self.loop_listener()
        self.after_loop()
        sleep(0.001*self.loop_interval_ms)
        __loop_wrapper()

    def __run_loop(self, run_loop_callback):
        # Do something
        __loop_wrapper()

    # For Neural Net

    def setDate():
        pass
