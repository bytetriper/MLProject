from transformers.tokenization_utils_base import BatchEncoding
import torch
class BatchEncodingWithPin(BatchEncoding):
    #__init__ receices a BatchEncoding object and return a BatchEncodingWithPin object
    #BatchEncodingWithPin is a subclass of BatchEncoding

    def __init__(self,be:BatchEncoding):
        super(BatchEncoding,self).__init__(be)
    def pin_memory(self):
        for k,v in self.items():
            if isinstance(v,torch.Tensor):
                self[k]=v.pin_memory()
        return self