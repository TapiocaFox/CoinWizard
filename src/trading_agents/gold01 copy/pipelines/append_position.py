from .pipeline import Pipeline
import copy
import torch
import torch.nn.functional as F

class AppendPositionPipeline(Pipeline):
    def __init__(self, position_length):
        super().__init__(torch.Tensor, torch.Tensor)
        self.position_length = position_length

    def setupInSpecs(self, in_specs):
        self.in_specs = in_specs
        self.device = in_specs['device']
        self.positions_tensor = torch.zeros((self.position_length, 3), device=self.device)
        self.out_specs = copy.deepcopy(self.in_specs)
        self.out_specs['tensor_feature_description_list'].append({'type': 'position', 'position': 'long'})
        self.out_specs['tensor_feature_description_list'].append({'type': 'position', 'position': 'wait'})
        self.out_specs['tensor_feature_description_list'].append({'type': 'position', 'position': 'short'})

    def generateOutSpecs(self):
        return self.out_specs

    def reset(self):
        raise NotImplementedError('Not implemented.')

    def toOneHot(self, x):
        return F.one_hot(x, num_classes=3)

    def pushPosistion(self, position):
        self.positions_tensor = self.positions_tensor.roll(-1, 0)
        self.positions_tensor[-1] = self.toOneHot(position)

    def _process(self, in_data, attachment_dict):
        return torch.cat((in_data, self.positions_tensor), dim=-1), attachment_dict

# Debug codes
# p = OhlcPandasToTorchPipeline(int, int)
# p.process(1, 2)
# print(OhlcPandasToTorch==Pipeline)
