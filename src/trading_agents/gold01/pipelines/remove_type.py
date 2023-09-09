from .pipeline import Pipeline
import copy, torch

class RemoveTypePipeline(Pipeline):
    def __init__(self, type):
        super().__init__(torch.Tensor, torch.Tensor)
        self.type = type

    def setupInSpecs(self, in_specs):
        self.in_specs = in_specs
        self.device = in_specs['device']
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

        self.in_type_to_be_removed_index_tensor = torch.tensor(self.in_type_to_be_removed_index_list, device=self.device)
        self.in_not_type_to_be_removed_index_tensor = torch.tensor(self.in_not_type_to_be_removed_index_list, device=self.device)

        # reverse
        for in_type_to_be_removed_index in self.in_type_to_be_removed_index_list[::-1]:
            del self.out_specs['tensor_feature_description_list'][in_type_to_be_removed_index]

    def generateOutSpecs(self):
        return self.out_specs

    def reset(self):
        raise NotImplementedError('Not implemented.')

    def _process(self, in_data, attachment_dict):
        out_data = in_data.index_select(-1, self.in_not_type_to_be_removed_index_tensor)
        return out_data, attachment_dict
