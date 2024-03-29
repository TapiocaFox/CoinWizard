#!/usr/bin/python3

import time, traceback, signal
from datetime import datetime

time_out_s = 15

def timeout_handler(signum, frame):
    raise Exception('BrokerEventLoopAPI loop timeout('+str(time_out_s)+' seconds passed).')

def dummp_func(BrokerAPI):
    pass

class BrokerEventLoopAPI(object):
    def __init__(self, before_loop, after_loop, broker_settings, nsp, loop_interval_ms= 150):
        self.before_loop = before_loop
        self.after_loop = after_loop
        self.notification_service_provider = nsp
        self.nsp = nsp
        self.latest_loop_datetime = None
        self.loop_listener = dummp_func
        self.every_15_second_listener = dummp_func
        self.latest_every_15_second_loop_datetime = None
        self.loop_interval_ms = loop_interval_ms
        self.stopped = True

    def order(self, instrument_name, order_settings, trade_settings):
        pass

    def getInstrument(self, instrument_name):
        pass

    def getAccount(self):
        return self.account

    def getNotificationServiceProvider(self):
        return self.nsp

    def onLoop(self, loop_listener):
        self.loop_listener = loop_listener

    def onEvery15Second(self, every_15_second_listener):
        self.every_15_second_listener = every_15_second_listener

    def _stop(self):
        if self.stopped:
            return
        self.stopped = True

    def _loop(self):
        pass

    def _loop_wrapper(self):
        # start_loop_timeStamp = datetime.now().timestamp()
        self.before_loop()
        self._loop()
        # *********** Need to be rewrite to fit syncing the clock. And timeout funciton
        # Fire every_15_second_listener if needed.
        if 1000*(datetime.now().timestamp() - self.latest_every_15_second_loop_datetime.timestamp()) >= 15000:
            self.every_15_second_listener(self)
            self.latest_every_15_second_loop_datetime = datetime.now()

        self.loop_listener(self)

        # Fire every_15_second_listener if needed.
        end_loop_timestamp = datetime.now().timestamp()
        time_passed_ms = (end_loop_timestamp - self.latest_loop_datetime.timestamp())*1000
        every_15_second_loop_remain_ms = 15000 - (end_loop_timestamp - self.latest_every_15_second_loop_datetime.timestamp())*1000
        # print('every_15_second_loop_remain_ms', every_15_second_loop_remain_ms)
        if every_15_second_loop_remain_ms <= (self.loop_interval_ms - time_passed_ms):
            if every_15_second_loop_remain_ms > 0:
                time.sleep(0.001*(every_15_second_loop_remain_ms))
            self.every_15_second_listener(self)
            self.latest_every_15_second_loop_datetime = datetime.now()
            end_loop_timestamp = datetime.now().timestamp()

        self.after_loop()

        time_passed_ms = (end_loop_timestamp - self.latest_loop_datetime.timestamp())*1000
        # print('time_passed_ms', time_passed_ms, self.loop_interval_ms)
        if(time_passed_ms < self.loop_interval_ms):
            # print(0.001*self.loop_interval_ms - time_passed_ms)
            time.sleep(0.001*(self.loop_interval_ms - time_passed_ms))

    def _run_loop(self):
        self.stopped = False
        self.latest_loop_datetime = datetime.now()
        self.latest_every_15_second_loop_datetime = datetime.now()
        loop_failed_count = 0
        signal.signal(signal.SIGALRM, timeout_handler)
        while True:
            if self.stopped:
                return
            try:
                signal.alarm(time_out_s)
                self._loop_wrapper()
                signal.alarm(0)
                loop_failed_count = 0
            except Exception as err:
                loop_failed_count += 1
                traceback.print_tb(err.__traceback__)
                print(err)
                print('A loop skipped with a exception. This is a '+str(loop_failed_count)+' times failure.')
                print('Sleep 5 seconds.')
                time.sleep(5)
                if loop_failed_count > 3:
                    print('Too many failures, skipped next loop.')
                    break
            self.latest_loop_datetime = datetime.now()
