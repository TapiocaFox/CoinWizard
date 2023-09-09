from .pipeline import Pipeline
from torch import Tensor
from pandas import DataFrame
import copy

def from_dataframe_to_tensor(instrument_list, primary_intrument, hist_data, device):
    data_list = []
    # print(hist_data)

    for i in range(len(instrument_list)):
        suffix = str(i)
        if primary_intrument==i:
            suffix=''
        # print(hist_data['open'+suffix])
        data_list.append(hist_data['open'+suffix].to_list())
        data_list.append(hist_data['high'+suffix].to_list())
        data_list.append(hist_data['low'+suffix].to_list())
        data_list.append(hist_data['close'+suffix].to_list())

    tensor = torch.tensor(data_list, device=device).transpose(1, 0)

    time_features_tensor = torch.from_numpy(time_features(hist_data.timestamp)).to(device)
    # print(time_features_tensor.type())
    # print(tensor.shape, time_features_tensor.shape)
    return torch.cat((tensor, time_features_tensor), dim=-1)

class AppendCulPipeline(Pipeline):
    def __init__(self):
        super().__init__(Tensor, Tensor)

    def setupInSpecs(self, in_specs):
        self.in_specs = in_specs
        self.instrument_list = in_specs['instrument_list']
        self.open_prefix = in_specs['open_prefix']
        self.high_prefix = in_specs['high_prefix']
        self.low_prefix = in_specs['low_prefix']
        self.close_prefix = in_specs['close_prefix']
        self.out_specs = copy.deepcopy(self.in_specs)
        tensor_feature_description_list = self.out_specs['tensor_feature_description_list']

        for i, instrument in enumerate(self.instrument_list):
            tensor_feature_description_list.append({'type': 'culr', 'instrument': instrument, 'culr': 'close'})
            tensor_feature_description_list.append({'type': 'culr', 'instrument': instrument, 'culr': 'upper'})
            tensor_feature_description_list.append({'type': 'culr', 'instrument': instrument, 'culr': 'lower'})
            tensor_feature_description_list.append({'type': 'culr', 'instrument': instrument, 'culr': 'realbody'})
            base_index = 4*i

        self.out_specs['tensor_feature_description_list'] = tensor_feature_description_list

    def generateOutSpecs(self):
        return self.out_specs

    def reset(self):
        raise NotImplementedError('Not implemented.')

    def _process(self, in_data, attachment_dict):
        culr_pandas = attachment_dict['ohlc_df']
        # print(culr_pandas)
        return in_data, attachment_dict

# Debug codes
# p = OhlcPandasToTorchPipeline(int, int)
# p.process(1, 2)
# print(OhlcPandasToTorch==Pipeline)
