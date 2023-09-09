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

def calculate_gamma(horizon, t_len):
    return 1 - (t_len/horizon)
