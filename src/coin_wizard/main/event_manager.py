#!/usr/bin/python3

import asyncio
import concurrent.futures
import functools

class EventManager(object):
    def __init__(self):
        self.event_listeners = {}
        self.stopped = False
        self.tasks = []

    # def start_loop(self):
    #     asyncio.run(self.loop())
    #
    # def stop_loop(self):
    #     self.stopped = True
    #
    # async def loop(self):
    #     while self.stopped:
    #         await asyncio.gather(*self.tasks)
    #         await asyncio.sleep(1)

    def on(self, event_name, listener):
        self.event_listeners[event_name] = listener
        # print(self.event_listeners)

    # async def emit_async(self, event_name, params):
    #     self.event_listeners[event_name](params)

    def emit(self, event_name, params):
        # self.tasks.append(asyncio.to_thread(self.event_listeners[event_name], params))
        return self.event_listeners[event_name](params)
