from .pipeline import Pipeline
from torch import Tensor
import copy

class RemoveTypePipeline(Pipeline):
    def __init__(self, type):
        super().__init__(Tensor, Tensor)
        self.type = type

    def setupInSpecs(self, in_specs):
        self.in_specs = in_specs
        self.in_type_to_be_removed_index_list = []
        self.in_not_type_to_be_removed_index_list = []
        self._init_specs(in_specs)

    def _init_specs(self, in_specs):
        tensor_feature_description_list = in_specs['tensor_feature_description_list']
        self.out_specs = copy.deepcopy(self.in_specs)

        for i, tensor_feature_description in enumerate(tensor_feature_description_list):
            if tensor_feature_description['type'] == self.type:
                self.in_type_to_be_removed_index_list.append(i)
            else:
                self.in_not_type_to_be_removed_index_list.append(i)

        # reverse
        for in_type_to_be_removed_index in self.in_type_to_be_removed_index_list[::-1]:
            del self.out_specs['tensor_feature_description_list'][in_type_to_be_removed_index]

    def generateOutSpecs(self):
        return self.out_specs

    def reset(self):
        raise NotImplementedError('Not implemented.')

    def _process(self, in_data, attachment_dict):
        return in_data, attachment_dict
