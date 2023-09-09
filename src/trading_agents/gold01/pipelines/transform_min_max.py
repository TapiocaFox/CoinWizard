from .pipeline import Pipeline
import copy
import torch

class TransformMinMaxPipeline(Pipeline):
    def __init__(self, start, stop, step):
        super().__init__(torch.Tensor, torch.Tensor)
        self.start = start
        self.stop = stop
        self.step = step

    def setupInSpecs(self, in_specs):
        self.in_specs = in_specs
        self.device = in_specs['device']
        self.out_specs = copy.deepcopy(self.in_specs)

    def generateOutSpecs(self):
        return self.out_specs

    def reset(self):
        raise NotImplementedError('Not implemented.')

    def _process(self, in_data, attachment_dict):
        start = self.start
        stop = self.stop
        step = self.step

        selected_data = in_data[:, start:stop:step]

        min = torch.min(selected_data, dim=0, keepdim=True)[0]
        max = torch.max(selected_data, dim=0, keepdim=True)[0]
        in_data[:, start:stop:step] = (selected_data-min)/(max-min)

        # print(min)
        # print(max)
        # print(selected_data)
        # print(in_data)
        # raise
        return in_data, attachment_dict

# Debug codes
# p = OhlcPandasToTorchPipeline(int, int)
# p.process(1, 2)
# print(OhlcPandasToTorch==Pipeline)
