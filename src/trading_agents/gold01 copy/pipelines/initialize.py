from .pipeline import Pipeline
from pandas import DataFrame
import copy
import torch

class InitializePipeline(Pipeline):
    def __init__(self, device):
        super().__init__(DataFrame, torch.Tensor)
        self.device = device

    def setupInSpecs(self, in_specs):
        self.in_specs = in_specs
        self.instrument_list = in_specs['instrument_list']
        self.open_prefix = in_specs['open_prefix']
        self.high_prefix = in_specs['high_prefix']
        self.low_prefix = in_specs['low_prefix']
        self.close_prefix = in_specs['close_prefix']
        self.out_specs = copy.deepcopy(self.in_specs)
        self.out_specs['tensor_feature_description_list'] = []
        self.out_specs['device'] = self.device

    def generateOutSpecs(self):
        return self.out_specs

    def reset(self):
        raise NotImplementedError('Not implemented.')

    def _process(self, in_data, attachment_dict):
        attachment_dict['ohlc_df'] = in_data
        observation_period = self.in_specs['observation_period']
        # print(self.in_specs['observation_period'])
        return torch.empty((observation_period, 0), device=self.device), attachment_dict

# Debug codes
# p = OhlcPandasToTorchPipeline(int, int)
# p.process(1, 2)
# print(OhlcPandasToTorch==Pipeline)
