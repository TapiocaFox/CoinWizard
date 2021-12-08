#!/usr/bin/python3
import sys
sys.path.append('../..')

import torch, random, math

import coin_wizard.historical_pair_data as hist
from datetime import datetime, timedelta
from time_features import time_features
time_delta_1_days = timedelta(days=7)


granularity_time_delta = {
    "M1": timedelta(seconds=60),
    "M5": timedelta(seconds=60*5),
    "M15": timedelta(seconds=60*15),
    "M30": timedelta(seconds=60*30),
    "H1": timedelta(seconds=60*60),
    "H4": timedelta(seconds=60*240),
    "D": timedelta(seconds=60*60*24),
}

def from_dataframe_to_tensor(hist_data, device):

    tensor = torch.tensor([
        hist_data.open.to_list(),
        hist_data.high.to_list(),
        hist_data.low.to_list(),
        hist_data.close.to_list()
    ], device=device).transpose(1, 0)

    time_features_tensor = torch.from_numpy(time_features(hist_data.timestamp)).to(device)
    # print(time_features_tensor.type())
    # print(tensor.shape, time_features_tensor.shape)
    return torch.cat((tensor, time_features_tensor), dim=-1)

class DataLoader(object):
    def __init__(self, instrument, granularity_list=None, primary_granularity=None, input_period_list=None, from_datetime=None, to_datetime=None, episode_steps=None, cuda=False):
        self.instrument = instrument
        self.primary_granularity = primary_granularity
        self.input_period_list = input_period_list
        self.granularity_list = granularity_list
        self.hist_data_granularities = [None for g in granularity_list]
        self.hist_data_tensor_granularities = [None for g in granularity_list]
        self.episode_steps = episode_steps
        self.device = torch.device('cuda:0') if cuda else torch.device('cpu')

        timestamp_series_granularities = [None for g in granularity_list]

        for granularity_index, granularity in enumerate(granularity_list):
            hist_data = hist.get_historical_pair_data_pandas(instrument, from_datetime, to_datetime, granularity=granularity, invert=False)
            hist_data = hist_data.dropna().reset_index()
            self.hist_data_granularities[granularity_index] = hist_data

            tensor = from_dataframe_to_tensor(hist_data, self.device)
            # print(tensor.type())

            # print(tensor[-1, 0].item())
            # print(hist_data)
            # print(tensor.shape)
            # print(hist_data)
            self.hist_data_tensor_granularities[granularity_index] = tensor
            timestamp_series_granularities[granularity_index] = hist_data.timestamp

        primary_timestamp = timestamp_series_granularities[primary_granularity]

        first_valid_timestamp = max([timestamp_series_granularities[granularity].iloc[input_period] for granularity, input_period in enumerate(input_period_list)])
        self.first_valid_index = primary_timestamp[primary_timestamp>=first_valid_timestamp].index[0]
        latest_valid_timestamp = min([timestamp_series_granularities[index].iloc[-1] for index, g in enumerate(granularity_list)])
        self.latest_valid_index = primary_timestamp[primary_timestamp>=latest_valid_timestamp].index[0]

        # print(self.first_valid_index)
        # # print(first_valid_timestamp)
        # print(self.latest_valid_index)
        # # print(latest_valid_timestamp)

        # Generate fast inference index list
        self.fast_inference_index_list = []

        first_valid_index = self.first_valid_index
        latest_valid_index = self.latest_valid_index
        hist_data_granularities = self.hist_data_granularities

        # Rolling through primary granularity dataframe
        for i in range(first_valid_index, len(hist_data_granularities[self.primary_granularity])):
            timestamp = (hist_data_granularities[self.primary_granularity].timestamp)[i]

            result = []
            for granularity_index, granularity in enumerate(granularity_list):
                hist_data = hist_data_granularities[granularity_index]
                timestamp_series = hist_data.timestamp
                index = timestamp_series.searchsorted(timestamp-granularity_time_delta[self.granularity_list[granularity_index]], side='right') - 1
                result.append(index)

            self.fast_inference_index_list.append(result)
        # print(self.fast_inference_index_list)

    def generateEpisode(self):
        primary_granularity = self.primary_granularity
        input_period_list = self.input_period_list
        granularity_list = self.granularity_list
        hist_data_granularities = self.hist_data_granularities
        hist_data_tensor_granularities = self.hist_data_tensor_granularities
        episode_steps = self.episode_steps
        first_valid_index = self.first_valid_index
        latest_valid_index = self.latest_valid_index


        random_index = random.randint(0, len(self.fast_inference_index_list)-self.episode_steps)
        # for g, i in enumerate(self.fast_inference_index_list[random_index]):
        #     print(hist_data_granularities[g].iloc[i])
        #     print(hist_data_tensor_granularities[g][i])


        return hist_data_tensor_granularities, self.fast_inference_index_list[random_index: random_index+self.episode_steps]
