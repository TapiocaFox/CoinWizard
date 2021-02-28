#!/usr/bin/python3
#

class Order(object):
    def __init__(self):
        self.filled_listener = None
        self.canceled_listener = None

    def cancel(self):
        pass

    def onFilled(self, filled_listener):
        self.filled_listener = filled_listener

    def onCanceled(self, canceled_listener):
        self.canceled_listener = canceled_listener
