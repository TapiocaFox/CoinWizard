#!/usr/bin/python3
#

class Account(object):
    def __init__(self, update_account):
        self.balance = 0.0
        self.currency = 'usd'
        self.margin_rate = 0.2
        self.margin_used = 0.0
        self.margin_available = 0.0
        self.unrealized_pl = 0.0
        self.orders = []
        self.trades = []
        self.update_account = update_account

    def getBalance(self):
        self.update_account()
        return self.balance

    def getCurrency(self):
        return self.currency

    def getMarginRate(self):
        return self.margin_rate

    def getMarginAvailable(self):
        self.update_account()
        return self.margin_available

    def getMarginUsed(self):
        self.update_account()
        return self.margin_used

    def getUnrealizedPL(self):
        self.update_account()
        return self.unrealized_pl

    def getOrders(self):
        return self.orders.copy()

    def getTrades(self):
        return self.trades.copy()
