from .pipeline import Pipeline

class PrinterPipeline(Pipeline):
    def __init__(self, in_out_type, print_attachment_dict=False):
        super().__init__(in_out_type, in_out_type)
        self.print_attachment_dict = print_attachment_dict

    def setupInSpecs(self, in_specs):
        self.in_specs = in_specs

    def generateOutSpecs(self):
        return self.in_specs

    def reset(self):
        raise NotImplementedError('Not implemented.')

    def _process(self, in_data, attachment_dict):
        print(in_data)
        if self.print_attachment_dict:
            print(attachment_dict)
        return in_data, attachment_dict

# Debug codes
# p = OhlcPandasToTorch(int, int)
# p.process(1, 2)
# print(OhlcPandasToTorch==Pipeline)
