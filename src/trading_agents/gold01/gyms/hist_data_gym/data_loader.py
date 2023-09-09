#!/usr/bin/python3
import sys
sys.path.append('../..')

import torch, random, math

import coin_wizard.historical_pair_data as hist
from datetime import datetime, timedelta


granularity_time_delta = {
    "M1": timedelta(seconds=60),
    "M5": timedelta(seconds=60*5),
    "M15": timedelta(seconds=60*15),
    "M30": timedelta(seconds=60*30),
    "H1": timedelta(seconds=60*60),
    "H4": timedelta(seconds=60*240),
    "D": timedelta(seconds=60*60*24),
}


class DataLoader(object):
    def __init__(self, instrument_list, primary_intrument, granularity, observation_period=None, from_datetime=None, to_datetime=None, episode_steps=None):
        self.instrument_list = instrument_list
        self.granularity = granularity
        self.observation_period = observation_period
        self.hist_data = None
        self.episode_steps = episode_steps
        # self.device = torch.device('cuda:0') if cuda else torch.device('cpu')

        timestamp_series = None

        hist_data = hist.get_historical_pair_data_pandas(instrument_list[primary_intrument], from_datetime, to_datetime, granularity=granularity, invert=False)
        hist_data = hist_data.dropna().reset_index(drop=True)
        for instrument_index, instrument in enumerate(instrument_list):
            if instrument_index == primary_intrument:
                continue
            instrument_hist_data = hist.get_historical_pair_data_pandas(instrument, from_datetime, to_datetime, granularity=granularity, invert=False)
            instrument_hist_data = instrument_hist_data.dropna().reset_index(drop=True)
            # Merge
            hist_data = hist_data.merge(instrument_hist_data, how='left', on='timestamp', suffixes=(None, str(instrument_index)))

        hist_data = hist_data.rename(columns={'open': 'open'+str(primary_intrument) ,'high': 'high'+str(primary_intrument), 'low': 'low'+str(primary_intrument), 'close': 'close'+str(primary_intrument)})
        hist_data = hist_data.fillna(0)

        self.hist_data = hist_data

        timestamp_series = hist_data.timestamp

        timestamp_series = timestamp_series

        first_valid_timestamp = timestamp_series.iloc[observation_period]
        self.first_valid_index = timestamp_series[timestamp_series>=first_valid_timestamp].index[0]
        latest_valid_timestamp = timestamp_series.iloc[-1]
        self.latest_valid_index = timestamp_series[timestamp_series>=latest_valid_timestamp].index[0]

        # print(self.first_valid_index)
        # # print(first_valid_timestamp)
        # print(self.latest_valid_index)
        # # print(latest_valid_timestamp)

        # Generate fast inference index list
        self.fast_inference_index_list = []

        # first_valid_index = self.first_valid_index
        # latest_valid_index = self.latest_valid_index

        # Rolling through primary granularity dataframe
        # for i in range(first_valid_index, len(hist_data)):
        #     timestamp = (hist_data.timestamp)[i]
        #
        #     timestamp_series = hist_data.timestamp
        #     index = timestamp_series.searchsorted(timestamp-granularity_time_delta[self.granularity_list[granularity_index]], side='right') - 1
        #
        #     self.fast_inference_index_list.append(index)
        # print(self.fast_inference_index_list)

    def generateEpisode(self):
        granularity = self.granularity
        observation_period = self.observation_period
        hist_data = self.hist_data
        episode_steps = self.episode_steps
        first_valid_index = self.first_valid_index
        latest_valid_index = self.latest_valid_index


        random_index = random.randint(first_valid_index, latest_valid_index-self.episode_steps)
        # for g, i in enumerate(self.fast_inference_index_list[random_index]):
        #     print(hist_data_granularities[g].iloc[i])
        #     print(hist_data_tensor_granularities[g][i])

        # result = hist_data.iloc[random_index:random_index].reset_index()

        return hist_data, random_index
