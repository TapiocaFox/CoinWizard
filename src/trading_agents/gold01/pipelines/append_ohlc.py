from .pipeline import Pipeline
from pandas import DataFrame
import copy, torch

def from_ohlc_dataframe_to_tensor(open_prefix, high_prefix, low_prefix, close_prefix, instrument_list, df, device):
    data_list = []

    for i in range(len(instrument_list)):
        suffix = str(i)
        data_list.append(df[open_prefix+suffix].to_list())
        data_list.append(df[high_prefix+suffix].to_list())
        data_list.append(df[low_prefix+suffix].to_list())
        data_list.append(df[close_prefix+suffix].to_list())

    return torch.tensor(data_list, device=device).transpose(1, 0)

class AppendOhlcPipeline(Pipeline):
    def __init__(self):
        super().__init__(torch.Tensor, torch.Tensor)

    def setupInSpecs(self, in_specs):
        self.in_specs = in_specs
        self.device = in_specs['device']
        self.instrument_list = in_specs['instrument_list']
        self.open_prefix = in_specs['open_prefix']
        self.high_prefix = in_specs['high_prefix']
        self.low_prefix = in_specs['low_prefix']
        self.close_prefix = in_specs['close_prefix']
        self.out_specs = copy.deepcopy(self.in_specs)
        tensor_feature_description_list = self.out_specs['tensor_feature_description_list']

        for i, instrument in enumerate(self.instrument_list):
            tensor_feature_description_list.append({'type': 'ohlc', 'instrument': instrument, 'ohlc': 'open'})
            tensor_feature_description_list.append({'type': 'ohlc', 'instrument': instrument, 'ohlc': 'high'})
            tensor_feature_description_list.append({'type': 'ohlc', 'instrument': instrument, 'ohlc': 'low'})
            tensor_feature_description_list.append({'type': 'ohlc', 'instrument': instrument, 'ohlc': 'close'})
            base_index = 4*i

        self.out_specs['tensor_feature_description_list'] = tensor_feature_description_list

    def generateOutSpecs(self):
        return self.out_specs

    def reset(self):
        raise NotImplementedError('Not implemented.')

    def _process(self, in_data, attachment_dict):
        ohlc_df = attachment_dict['ohlc_df']
        # print(ohlc_pandas)
        out_data = from_ohlc_dataframe_to_tensor(self.open_prefix, self.high_prefix, self.low_prefix, self.close_prefix, self.instrument_list, ohlc_df, self.device)

        return torch.cat((in_data, out_data), dim=-1), attachment_dict
