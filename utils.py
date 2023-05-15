from transformers.tokenization_utils_base import BatchEncoding
import torch
import numpy as np
import PIL.Image as Image
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    make_list_of_images)
from typing import Union,List
from Params import Params
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import CLIPTokenizerFast
import torch
import datasets
class Img_Processor():
    """
        deprecated
    """
    def __init__(self, img_size: int = 224, Rescale_Factor: int = 255, device: torch.device = torch.device('cpu')):
        self.img_size = img_size
        self.Rescale_Factor = Rescale_Factor
        self.device = device
    def __call__(self, img: Union[Image.Image, List[Image.Image]], do_rescale: bool = True, convert_rgb: bool = True) -> torch.Tensor:
        listed_img = make_list_of_images(img)
        if convert_rgb:
            listed_img = [img.convert("RGB") for img in listed_img]
        listed_img = [img.resize((224, self.img_size))
                      for img in listed_img]
        listed_img = [torch.Tensor(np.array(img, dtype=float)).permute(
            2, 0, 1) for img in listed_img]
        if do_rescale:
            listed_img = [img/self.Rescale_Factor for img in listed_img]
        img = torch.stack(listed_img).to(self.device)
        return img
class Gen_To_Clip_Processor():
    """
        pre-process the image to be compatible with CLIP
        args:
            mean: mean of the CLIP model by default
            std: std of the CLIP model by default
        bahaviour:
            normalize the image using the mean and std
            unsqueeze the image if it is of shape (channel,height,width) to (batch_size,channel,height,width)
    """
    def __init__(self, mean: list = OPENAI_CLIP_MEAN, std: list = OPENAI_CLIP_STD) -> None:
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # normalize the testimg using the mean and std
        # img: (batch_size,3,224,224)
        # if img is of shape (3,224,224), then add a dimension
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        for channel in range(3):
            img[:, channel] = (
                img[:, channel]-self.mean[channel])/self.std[channel]
        return img
class CLIPLoader(DataLoader):
    """
    This loader inherit every method from DataLoader, but the collate_fn is changed to adapt to CLIP
    """
    @property
    def tokenizer_default(self):
        if not hasattr(self,'_tokenizer_default'):
            self._tokenizer_default=CLIPTokenizerFast.from_pretrained(Params["model_name"],local_files_only=True)
        return self._tokenizer_default
    def collate_fn_default(self,x:list[dict])->BatchEncoding:
        # x: [{img:tensor,caption:{input_ids: list[int],attention_mask:list[int]} },...]
        # return a BatchEncoding
        batched_data = self.tokenizer_default([i['caption'][0][:100]
                                 for i in x], padding=True, return_tensors='pt')
        batched_data['pixel_values'] = torch.stack(
            [i['image'].float()/255 for i in x])
        return batched_data
    def __init__(self,dataset:Dataset,batch_size:int=1,shuffle:bool=False,num_workers:int=0,collate_fn=None,pin_memory:bool=False,pin_memory_device=None):
        if collate_fn is None:
            collate_fn=self.collate_fn_default    
        super(CLIPLoader,self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            pin_memory_device=pin_memory_device
        )

def load_data_with_retry(path: str,keep_in_memory:bool=True, retry_times: int = 5) -> datasets.Dataset:
    """
    load data from disk, if it fails, retry for retry_times times
    """
    for i in range(retry_times):
        try:
            dataset = datasets.load_from_disk(path, keep_in_memory=keep_in_memory)
            return dataset
        except Exception as e:
            print(f"load data from {path} failed, retrying {i+1} times")
            print(e)
    raise Exception(f"load data from {path} failed after {retry_times} times")