#!/usr/bin/python3

import finplot as fplt
from dateutil.tz import gettz
from coin_wizard.technical_indicators import TechnicalIndicators

title = ''
candles = None
hover_label = None
ti = TechnicalIndicators()

red = '#b71c1c'
green = '#1b5e20'

def update_legend_text(x, y):
    global hover_label, candles
    row = candles.loc[candles.timestamp==x]
    # print(row, x)
    # format html with the candle and set legend
    fmt = '<span style="color:#%s">%%.5f</span>' % ('0b0' if (row.open<row.close).all() else 'a00')
    rawtxt = '<span style="font-size:13px">%%s</span> &nbsp; O: %s H: %s L: %s C: %s' % (fmt, fmt, fmt, fmt)
    hover_label.setText(rawtxt % (title, row.open, row.high, row.low, row.close))

def update_crosshair_text(x, y, xtext, ytext):
    global candles
    row = candles.iloc[x]
    # print(row, x)
    ytext += ' O: %.5f H: %.5f L: %.5f C: %.5f' % (row['open'], row['high'], row['low'], row['close'])
    return xtext, ytext

def plot_candles(title_, candles_, target_timezone='UTC'):
    global hover_label, candles, title
    title = title_
    candles = candles_
    ax, ax2 = fplt.create_plot(title, rows=2)
    candles['timestamp'] = candles['timestamp'].astype('int64')

    # rsi = ti.rsi_ema(candles.close)
    # fplt.plot(rsi, ax=ax3, legend='RSI')

    # print(candles['timestamp'])
    # print(candles)
    # print(candles[['timestamp', 'open', 'close', 'high', 'low']])
    macd = ti.macd(candles.close, short=12, long=26)
    signal = ti.ema(macd, period=9)
    candles['macd_diff'] =  macd - signal
    momentum = ti.momentum(candles.close)
    # print(candles[['timestamp', 'open', 'close', 'high', 'low', 'macd_diff']])

    fplt.volume_ocv(candles[['timestamp','open','close','macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
    fplt.plot(macd, ax=ax2, legend='MACD')
    fplt.plot(signal, ax=ax2, legend='Signal')
    fplt.plot(momentum, ax=ax2, legend='Monentum')


    # # change to b/w coloring templates for next plots
    # fplt.candle_bull_color = fplt.candle_bear_color = '#000'
    # fplt.volume_bull_color = fplt.volume_bear_color = '#333'
    # fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#fff'
    # ma = ti.ma(candles.close, 30)
    upper, lower = ti.bb(candles.close)

    fplt.display_timezone = gettz(target_timezone)
    fplt.candlestick_ochl(candles[['timestamp', 'open', 'close', 'high', 'low']], ax=ax)
    # fplt.plot(ma, ax=ax, legend='MA', color=green)
    fplt.plot(upper, ax=ax, legend='BB Upper', color=red)
    fplt.plot(lower, ax=ax, legend='BB Lower', color=green)
    hover_label = fplt.add_legend('', ax=ax)

    fplt.background = '#263238'
    fplt.odd_plot_background = '#263238'
    fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
    fplt.add_crosshair_info(update_crosshair_text, ax=ax)
    fplt.show()
