#!/usr/bin/python3

class Order(object):
    def onFilled(self, filled_listener):
        pass
    def onRejected(self, rejected_listener):
        pass
    def cancel(self):
        pass
    def __emit_filled(self):
        pass
    def __emit_rejected(self):
        pass
