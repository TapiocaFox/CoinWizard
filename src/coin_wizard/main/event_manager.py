#!/usr/bin/python3

import asyncio
import concurrent.futures
import functools

class EventManager(object):
    def __init__(self):
        self.event_listeners = {}

    def on(self, event_name, listener):
        self.event_listeners[event_name] = listener
        # print(self.event_listeners)

    async def emit_async(self, event_name, params):
        # print(self.event_listeners)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, functools.partial(self.event_listeners[event_name], params))
        return result

    def emit(self, event_name, params):
        asyncio.run(self.emit_async(event_name, params))
