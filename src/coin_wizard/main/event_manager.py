#!/usr/bin/python3

class EventManager(object):
    def __init__(self):
        self.event_listeners = {}

    def on(self, event_name, listener):
        self.event_listeners[event_name] = listener
        print(self.event_listeners)

    def emit(self, event_name, params):
        print(self.event_listeners)
        self.event_listeners[event_name](params)
