# CoinWizard
![alt text](/imgs/terminal.png)
A foreign currencies/forex trading management program.

# Installation
```sh
pip install pytz numpy pandas finplot bs4 prompt_toolkit dateutil pushbullet.py
```
# Start CoinWizard
go to "src" folder and
```sh
python3 start.py
```
# TradingAgent BrokerAPI Documentation
Index:
1. BrokerAPI
2. Account
3. Order
4. Trade
5. Instrument
6. NotificationServiceProvider
7. coin_wizard.historical_pair_data

## 1. BrokerAPI
### BrokerAPI.getAccount
Get the account object.
``` python3
account = BrokerAPI.getAccount()
```
### BrokerAPI.getInstrument
Get the instrument object.
``` python3
# Get the EUR_USD instrument.
instrument = BrokerAPI.getInstrument('EUR_USD')
```
### BrokerAPI.order
Order for an instrument with order and trade settings.
``` python3
# Order 2 units EUR_USD instrument in the market immediately.
order = BrokerAPI.order('EUR_USD', {"type": "market"}, {"units": 2})
```
``` python3
# Order stop order of EUR_USD instrument. With load of settings.
order = BrokerAPI.order('EUR_USD', {"type": "stop", "price": 2, "bound": 2.1}, {"units": 1, "take_profit": 2, "stop_lost": 0.5, "trailing_stop_distance": 0.001})
```
### BrokerAPI.getNotificationServiceProvider
Get the notification_service_provider object.
``` python3
nsp = BrokerAPI.getNotificationServiceProvider()
```
### BrokerAPI.onLoop
Register listener/callback for every loop of event-loop.
``` python3
# Note that every times that the callback emitted BrokerAPI is passed as parameters.
def loop(BrokerAPI):
  # do something here
  pass

# With every loop of event-loop call loop() function.
order = BrokerAPI.onLoop(loop)
```
### BrokerAPI.onEvery15Second
Register listener/callback for every 15 seconds.
``` python3
# Note that every times that the callback emitted BrokerAPI is passed as parameters.
def loop15s(BrokerAPI):
  # do something here
  pass

# With every 15 seconds of event-loop call loop15s() function.
order = BrokerAPI.onEvery15Second(loop15s)
```
## 2. Account
First, get the account object from BrokerAPI.
``` python3
account = BrokerAPI.getAccount()
```
### Account.getBalance
Get your account balance in float type.
``` python3
balance = account.getBalance()
# For example 10000.0
```
### Account.getCurrency
Get your account balance in string type.
``` python3
currency = account.getCurrency()
# For example "USD"
```
### Account.getMarginRate
Get your account margin rate in float type.
``` python3
margin_rate = account.getMarginRate()
# For example 0.02
```
### Account.getMarginAvailable
Get your account margin available in float type.
``` python3
margin_available = account.getMarginAvailable()
# For example 99980.0
```
### Account.getMarginUsed
Get your account margin used in float type.
``` python3
margin_used = account.getMarginUsed()
# For example 20.0
```
### Account.getUnrealizedPL
Get your account unrealized profit&loss in float type.
``` python3
unrealized_pl = account.getUnrealizedPL()
# For example -20.0
```
### Account.getOrders
Get your account orders in list.
``` python3
orders = account.getOrders()
# For example [order_object_1, order_object_2]
```
### Account.getTrades
Get your account trades in list.
``` python3
trades = account.getTrades()
# For example [trade_object_1]
```
## 3. Order
First, you should get your order either from BrokerAPI.order(...params) or Account.getOrders()
``` python3
# Order 2 units EUR_USD instrument in the market immediately.
order = BrokerAPI.order('EUR_USD', {"type": "market"}, {"units": 2})
```
``` python3
orders = account.getOrders()
# For example [order_object_1, order_object_2]
```
### Order.onFilled
Register listener/callback for order filled event.
``` python3
# Note that every times that the callback emitted order and trade is passed as parameters.
def filled_listener(order, trade):
  # do something here
  pass

# When the order filled, call "filled_listener".
order.onFilled(filled_listener)
```
### Order.onCanceled
Register listener/callback for order filled event.
``` python3
# Note that every times that the callback emitted order and reason(string type) is passed as parameters.
def canceled_listener(order, reason):
  # do something here
  pass

# When the order canceled, call "canceled_listener".
order.onCanceled(canceled_listener)
```
### Order.cancel
Cancel the order itself.
``` python3
order.cancel()
```
Do not forget to register order canceled listener.
``` python3
order.onCanceled(canceled_listener)
```
### Order.getInstrumentName
Get your order instrument name.
``` python3
instrument_name = order.getInstrumentName()
# For example "EUR_USD"
```
### Order.getOrderSettings
Get your order's order settings.
``` python3
order_settings = order.getOrderSettings()
# For example
# {
#  "type": "market"
# }
```
### Order.getTradeSettings
Get your order's trade setting.
``` python3
trade_settings = order.getTradeSettings()
# For example
# {
#  "units": -7997.0,
#  "current_units": -7996.0,
#  "take_profit": 0.543,
#  "stop_loss": 2.0,
#  "trailing_stop_distance": 0.1
# }
```
## 4. Trade
First, you should get your order either from order.onFilled or Account.getTrades()
``` python3
def filled_listener(order, trade):
  # You get your trade of that order here.
  pass

# When the order filled, call "filled_listener".
order.onFilled(filled_listener)
```
``` python3
trades = account.getTrades()
# For example [trade_object_1, trade_object_2]
```
### Trade.onClosed
Register listener/callback for trade closed event.
``` python3
# Note that every times that the callback emitted.
# trade, realized_pl, close_price, spread, timestamp is passed as parameters.
def closed_listener(self, trade, realized_pl, close_price, spread, timestamp):
  # do something here
  pass

# When the trade closed, call "closed_listener".
trade.onClosed(closed_listener)
```
### Trade.onReduced
Register listener/callback for trade closed event.
``` python3
# Note that every times that the callback emitted.
# trade, units, realized_pl, close_price, spread, timestamp is passed as parameters.
# Different from Trade.onClosed, Trade.onReduced pass units parameters. For example: units = -1.0, units = 2.0.
def reduced_listener(trade, units, realized_pl, close_price, spread, timestamp):
  # do something here
  pass

# When the trade reduced, call "reduced_listener".
trade.onReduced(reduced_listener)
```
### Trade.close
Close the trade itself.
``` python3
trade.close()
```
Do not forget to register trade closed listener.
``` python3
trade.onClosed(closed_listener)
```
### Trade.reduce
Reduce 10 units of long position trade. Both accepted.
``` python3
trade.reduce(-10)
```
``` python3
trade.reduce(10)
```
Reduce 10 units of short position.
``` python3
trade.reduce(10)
```
Do not forget to register trade closed listener.
``` python3
trade.onClosed(closed_listener)
```
### Trade.getInstrumentName
Get your trade instrument name.
``` python3
instrument_name = trade.getInstrumentName()
# For example "EUR_USD"
```
### Trade.getOpenPrice
Get your trade open price.
``` python3
price = trade.getOpenPrice()
# For example 1.4325
```
### Trade.getTradeSettings
Get your trade's trade setting.
``` python3
trade_settings = trade.getTradeSettings()
# For example
# {
#  "units": -7997.0,
#  "current_units": -7996.0,
#  "take_profit": 0.543,
#  "stop_loss": 2.0,
#  "trailing_stop_distance": 0.1
# }
```
### Trade.getUnrealizedPL
Get your trade unrealized profit&loss.
``` python3
unrealized_pl = trade.getUnrealizedPL()
# For example -21.011
```
## 5. Instrument
First, get the account object from BrokerAPI.
``` python3
# Get the EUR_USD instrument.
instrument = BrokerAPI.getInstrument('EUR_USD')
```

### Instrument.getCurrentCloseoutBidAsk
Get instrument current closeout bid, ask and timestamp.
``` python3
bid, ask, timestamp = instrument.getCurrentCloseoutBidAsk()
# For example: 1.19289, 1.19303, datetime.datetime(2021, 3, 15, 12, 15, 29, 271891, tzinfo=tzutc())
```
### Instrument.getActive1MCandle
Get instrument active 1m candle dataframe(pandas).
``` python3
the_candle_df = instrument.getActive1MCandle()
# Return pandas dataframe with below fields:
# timestamp, open, high, low, close
# Note that timestamp is in utc iso format
```

### Instrument.getRecent1MCandles
Get instrument active recent 1m candle dataframe(pandas).
``` python3
# Get 1000 candles.
candles_df = instrument.getRecent1MCandles(1000)
# Return pandas 1000 rows dataframe with below fields:
# timestamp, open, high, low, close
# Note that timestamp is in utc iso format
```
### Instrument.isTradable
Check if such instrument is tradable.
``` python3
tradable = instrument.isTradable()
# For example: False
```
## 6. NotificationServiceProvider
First, get the notification_service_provider object from BrokerAPI.
``` python3
nsp = BrokerAPI.getNotificationServiceProvider()
```
### NotificationServiceProvider.pushImmediately()
Push a notification immediately.
``` python3
# Push notification with title and context.
nsp.pushImmediately('Title1', 'Hello')
```
### NotificationServiceProvider.addLine()
Add a notification line for later push. (Not yet sent/push)
``` python3
# Add a line.
nsp.addLine('Hello')
```
### NotificationServiceProvider.push()
Push accumulated notification lines with title.
``` python3
# Add a line.
nsp.addLine('Hello')
# Push.
nsp.push('Title1')
```
## 7. coin_wizard.historical_pair_data
In your trading agent
``` python3
import coin_wizard.historical_pair_data as historical_pair_data
```
### historical_pair_data.get_historical_pair_data
Get your historical data in numpy format with utc timestamp from 1970, 1, 1.
``` python3
# Get historical data in with eastern time setup.
historical_pair_data.get_historical_pair_data('eurusd', datetime(2021, 1, 8, 0, 0), datetime(2021, 1, 11, 23, 59))
```
``` python3
# Get historical data in with eastern time setup.
import pytz
eastern = pytz.timezone('US/Eastern')
historical_pair_data.get_historical_pair_data('eurusd', eastern.localize(datetime(2021, 1, 8, 0, 0)), eastern.localize(datetime(2021, 1, 11, 23, 59)))
```

### historical_pair_data.get_historical_pair_data_pandas
Get your historical data in pandas format. With default UTC timezone.
``` python3
# Get historical data in with eastern time setup.
historical_pair_data.get_historical_pair_data_pandas('eurusd', datetime(2021, 1, 8, 0, 0), datetime(2021, 1, 11, 23, 59))
```
``` python3
# Get historical data in with eastern time setup. And too output with timezone 'US/Eastern' as the latest parameter.
import pytz
eastern = pytz.timezone('US/Eastern')
historical_pair_data.get_historical_pair_data_pandas('eurusd', eastern.localize(datetime(2021, 1, 8, 0, 0)), eastern.localize(datetime(2021, 1, 11, 23, 59)), 'US/Eastern')
```
### historical_pair_data.plot_historical_pair_data
Plot your historical data.
``` python3
# Plot historical data in with eastern time setup.
import pytz
eastern = pytz.timezone('US/Eastern')
historical_pair_data.plot_historical_pair_data('eurusd', eastern.localize(datetime(2021, 1, 8, 0, 0)), eastern.localize(datetime(2021, 1, 11, 23, 59)), 'US/Eastern')
```
