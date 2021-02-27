#!/usr/bin/python3
#

class Order(object):
    def __init__(self):
        pass

    def onFilled(self, filled_listener):
        pass

    def onCanceled(self, canceled_listener):
        pass

    def cancel(self):
        pass

    def __emit_filled(self):
        pass

    def __emit_rejected(self):
        pass
