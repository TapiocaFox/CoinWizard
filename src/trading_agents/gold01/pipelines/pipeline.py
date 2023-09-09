

class Pipeline:
    def __init__(self, in_type, out_type):
        self.in_type = in_type
        self.out_type = out_type

    def _process(self, in_data, attachment_dict):
        raise NotImplementedError('Not implemented.')
        # return

    def setupInSpecs(self, in_specs):
        raise NotImplementedError('Not implemented.')

    def generateOutSpecs(self):
        raise NotImplementedError('Not implemented.')

    def reset(self):
        raise NotImplementedError('Not implemented.')
        # return

    def process(self, data, attachment_dict):
        if not isinstance(data, self.in_type):
            raise TypeError('Expect type "%s" as pipeline\'s input. Got "%s".'%(str(self.in_type), str(type(data))))
        out, attachment_dict = self._process(data, attachment_dict)
        if not isinstance(out, self.out_type):
            raise TypeError('Expect type "%s" as pipeline\'s output. Got "%s".'%(str(self.out_type), str(type(out))))
        return out, attachment_dict

# Debug codes
# p = Pipeline()
# p.process(1, 2)
# print(Pipeline==Pipeline)
