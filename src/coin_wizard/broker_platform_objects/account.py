#!/usr/bin/python3
#

class Account(object):
    def __init__(self):
        self.balance = 0.0
        self.currency = 'usd'
        self.margin_rate = 0.2
        self.margin_used = 0.0
        self.margin_available = 0.0
        self.unrealized_pl = 0.0
        self.orders = []
        self.trades = []

    def getBalance(self):
        return self.balance

    def getCurrency(self):
        return self.currency

    def getMarginRate(self):
        return self.margin_rate

    def getMarginUsed(self):
        return self.margin_used

    def getUnrealizedPL(self):
        return self.unrealized_pl

    def getOrders(self):
        return self.orders

    def getTrades(self):
        return self.trades
