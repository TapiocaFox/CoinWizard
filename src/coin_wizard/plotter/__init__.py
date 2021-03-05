#!/usr/bin/python3

import finplot as fplt
from dateutil.tz import gettz

title = ''
candles = None
hover_label = None

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

def plot_candles(title_, candles_, target_timezone):
    global hover_label, candles, title
    title = title_
    candles = candles_
    ax, ax2 = fplt.create_plot(title, rows=2)
    candles['timestamp'] = candles['timestamp'].astype('int64')
    # print(candles['timestamp'])
    # print(candles)
    # print(candles[['timestamp', 'open', 'close', 'high', 'low']])
    macd = candles.close.ewm(span=12).mean() - candles.close.ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    candles['macd_diff'] = macd - signal
    # print(candles[['timestamp', 'open', 'close', 'high', 'low', 'macd_diff']])

    fplt.volume_ocv(candles[['timestamp','open','close','macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
    fplt.plot(macd, ax=ax2, legend='MACD')
    fplt.plot(signal, ax=ax2, legend='Signal')

    # # change to b/w coloring templates for next plots
    # fplt.candle_bull_color = fplt.candle_bear_color = '#000'
    # fplt.volume_bull_color = fplt.volume_bear_color = '#333'
    # fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#fff'

    fplt.display_timezone = gettz(target_timezone)
    fplt.candlestick_ochl(candles[['timestamp', 'open', 'close', 'high', 'low']], ax=ax)
    hover_label = fplt.add_legend('', ax=ax)

    fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
    fplt.add_crosshair_info(update_crosshair_text, ax=ax)
    fplt.show()
