import pytz
import json
from datetime import datetime

import matplotlib.pyplot as plt

import torch, jsbeautifier, json
import torch.nn.functional as F

utc = pytz.utc

def pretty_print(x):
    options = jsbeautifier.default_options()
    options.indent_size = 2
    options.wrap_line_length = 100
    print(jsbeautifier.beautify(json.dumps(x), options))

def plot_list(list):
    x = [i+1 for i in range(len(list))]
    plt.plot(x, list, label='list')
    plt.title(str(123))
    plt.legend()
    plt.show()

def plot_tensor(tensor_list, use_points=[]):
    fig, ax_list = plt.subplots(len(tensor_list))
    if len(tensor_list) == 1:
        ax_list = [ax_list]
    for i, tensor in enumerate(tensor_list):
        ax = ax_list[i]
        length = tensor.shape[0]
        dim = tensor.shape[1]
        x = [i+1 for i in range(length)]
        for d in range(dim):
            if i in use_points:
                ax.plot(x, tensor[:, d].tolist(), 'o', markersize=2, label=str(d))
            else:
                ax.plot(x, tensor[:, d].tolist(), label=str(d))
        ax.set_title(str(tensor.shape))
        ax.legend()
    plt.show()
    # plt.show(block=False)
    # plt.pause(0.0001)
    # plt.close()

class ConfigLoader(object):
    def __init__(self, config_path):
        with open(config_path) as json_file:
            self.config = json.load(json_file)

    def get(self, key):
        return self.config[key]

    def getDate(self, *args):
        args = list(args)
        for i, arg in enumerate(args):
            args[i] = self.config[arg]
        return utc.localize(datetime(*args))

    # def getDictionary(self):
    #     return self.config

class InputDataPreProcessor:
    def __init__(self, concurrent_episodes, instrument_list, primary_intrument, data_length=96, recent_log_returns_length=8, normalize_length=96, cuda=False):
        if normalize_length < data_length:
            raise
        self.concurrent_episodes = concurrent_episodes
        self.instrument_list = instrument_list
        self.primary_intrument = primary_intrument
        self.data_length = data_length
        self.normalize_length = normalize_length
        self.recent_log_returns_length = recent_log_returns_length
        self.require_raw_data_length = data_length+normalize_length+recent_log_returns_length
        self.device = torch.device('cuda:0') if cuda else torch.device('cpu')
        self.positions = torch.zeros((self.concurrent_episodes, self.data_length, 3), device=self.device)
        self.positions[:, :, 1] = 1

    def reset(self):
        self.positions = torch.zeros((self.concurrent_episodes, self.data_length, 3), device=self.device)
        self.positions[:, :, 1] = 1
        
    def patchAllPositions(self, x):
        recent_log_returns_tensor = self.patchRecentLogReturns(x).unsqueeze(1).repeat(1, 3, 1, 1)
        positions = self.positions.unsqueeze(1).repeat(1, 3, 1, 1).clone()

        positions[:, 0, -1, :] = 0
        positions[:, 0, -1, 0] = 1
        positions[:, 1, -1, :] = 0
        positions[:, 1, -1, 1] = 1
        positions[:, 2, -1, :] = 0
        positions[:, 2, -1, 2] = 1

        # print(positions.shape)
        # print(recent_log_returns_tensor.shape)

        result = torch.cat((recent_log_returns_tensor, positions), dim=-1)
        return result

    def patchRecentLogReturns(self, x):
        raw_data_length = x.shape[1]
        # if raw_data_length < self.require_raw_data_length:
        #     raise
        close_index = 3
        close_index_tensor = x[:, :, close_index:4*len(self.instrument_list):4]
        time_features_tensor = x[:, -self.data_length:, 4*len(self.instrument_list):]


        recent_log_returns_tensor = torch.log(close_index_tensor[:, 1:, :]/close_index_tensor[:, :-1, :])
        # print(recent_log_returns_tensor.shape)
        recent_log_returns_tensor_norm = recent_log_returns_tensor.unfold(1, self.normalize_length, 1)
        # print(recent_log_returns_tensor_norm.shape)

        std = torch.std(recent_log_returns_tensor_norm, -1, keepdim=False)
        mean = torch.mean(recent_log_returns_tensor_norm, -1, keepdim=False)
        # print(std.shape)
        # print(recent_log_returns_tensor[:, self.normalize_length-1:, :].shape)

        recent_log_returns_tensor = (recent_log_returns_tensor[:, self.normalize_length-1:, :]-mean)/std
        # print(recent_log_returns_tensor)

        recent_log_returns_tensor = recent_log_returns_tensor.unfold(1, self.recent_log_returns_length, 1)[:, -self.data_length:, :, :]
        recent_log_returns_tensor = recent_log_returns_tensor.contiguous().view(self.concurrent_episodes, self.data_length, self.recent_log_returns_length*len(self.instrument_list))
        # print(recent_log_returns_tensor)
        # print(recent_log_returns_tensor.shape)
        recent_log_returns_tensor = torch.nan_to_num(recent_log_returns_tensor, nan=0 , posinf=0, neginf=0)

        result = torch.cat((time_features_tensor, recent_log_returns_tensor), dim=-1)
        return result

    def process(self, x):
        result = torch.cat((self.patchRecentLogReturns(x), self.positions), dim=-1)
        return result

    def pushPosition(self, new_position):
        new_position = self.toOneHot(new_position.long())
        self.positions = torch.roll(self.positions, -1, dims=1)
        self.positions[:, -1, :] = new_position

    def toOneHot(self, x):
        return F.one_hot(x, num_classes=3)

def calculate_gamma(horizon, t_len):
    return 1 - (t_len/horizon)
