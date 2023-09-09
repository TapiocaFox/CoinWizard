
class AccountBook:
    def __init__(self):
        self.trades_list = []

    def reset(self):
        self.trades_list = []

    def appendTrade(self, trade):
        self.trades_list.append(trade)

    def generateSummary(self):
        return
