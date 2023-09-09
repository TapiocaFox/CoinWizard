

class PipelineArray:
    def __init__(self, pipeline_list, first_pipeline_in_specs):
        self.pipeline_list = pipeline_list
        self.__validate_pipeline_list()
        self.out_specs = self.__init_pipeline_list(first_pipeline_in_specs)

    def __validate_pipeline_list(self):
        first = True
        predecessor_out_type = None
        for pipeline in self.pipeline_list:
            this_in_type = pipeline.in_type
            if first:
                first = False
            else:
                if predecessor_out_type!=this_in_type:
                    raise Exception('Pipeline array has mismatch in and out types "%s","%s" .'%(str(this_in_type), str(predecessor_out_type)))
            predecessor_out_type = pipeline.out_type

    def __init_pipeline_list(self, first_pipeline_in_specs):
        in_specs = first_pipeline_in_specs
        for pipeline in self.pipeline_list:
            pipeline.setupInSpecs(in_specs)
            in_specs = pipeline.generateOutSpecs()

        return in_specs

    def returnOutSpecs(self):
        return self.out_specs

    def process(self, data):
        attachment_dict = {}
        for pipeline in self.pipeline_list:
            try:
                data, attachment_dict = pipeline.process(data, attachment_dict)
            except Exception as e:
                print('Encounter error with pipeline "%s".'%(str(pipeline)))
                raise e

        return data, attachment_dict

    def resetAll(self):
        for pipeline in self.pipeline_list:
            pipeline.reset()
