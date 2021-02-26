import pytz
from datetime import datetime
import numpy as np

eastern = pytz.timezone('US/Eastern')

print((eastern.localize(datetime.strptime(str(b'20000530 172700', 'utf-8'), '%Y%m%d %H%M%S'))).timestamp())

print(datetime.utcfromtimestamp(959722020.0))

with open('pair_data/eurusd/DAT_ASCII_EURUSD_M1_2000.npy', 'rb') as f:
    a = np.load(f)

print(datetime.utcfromtimestamp(a[0][0]))

print(datetime.timestamp(datetime(1970, 1, 1, 0, 0)))
