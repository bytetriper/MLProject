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
from Constants import *
from torchvision import transforms
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
    def collate_fn_default(self,x:List[dict])->torch.Tensor:
        # x: [{img:tensor,caption:{input_ids: list[int],attention_mask:list[int]} },...]
        # return a BatchEncoding
        data= torch.stack(
            [i['image'].float()/255 for i in x])
        #batched_data= BatchEncoding(data={'pixel_values':data},return_tensors='pt')
        return data
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
class Image_Net_Constants():
    def __init__(self) -> None:
        self.mean=torch.tensor(IMAGENET_MEAN)
        self.std=torch.tensor(IMAGENET_STD)
    def __call__(self,x)->torch.Tensor:
        assert len(x.shape)==4
        returned=torch.zeros_like(x)
        for channel in range(3):
            returned[:,channel]=(x[:,channel]-self.mean[channel])/self.std[channel]
        return returned
class ImageNet_Loader(DataLoader):
    """
    A DataLoader for ImageNEt dataset
    """
    def __init__(self,noised:bool=True,batch_size:int=1,shuffle:bool=False,num_workers:int=0,collate_fn=None,pin_memory:bool=False,pin_memory_device=None):
        if collate_fn is None:
            collate_fn=self.collate_fn_default
        path=Params['train_dataset_path'] if not noised else Params['noised_dataset_path']
        dataset=datasets.load_from_disk(path,keep_in_memory=True)
        if noised:
            dataset.set_format(type='torch')
            collate_fn=None

        super(ImageNet_Loader,self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            pin_memory_device=pin_memory_device
        )
    def collate_fn_default(self,x:List[dict])->torch.Tensor:
        # x: [{img:tensor,caption:{input_ids: list[int],attention_mask:list[int]} },...]
        # return a BatchEncoding
        transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])
        data= torch.stack(
            [transform(i['image'].convert('RGB')).float() for i in x])
        label=torch.tensor(
            [i['label'] for i in x]
        )
        #batched_data= BatchEncoding(data={'pixel_values':data},return_tensors='pt')
        return {'image':data,'label':label}

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
from matplotlib import pyplot as plt
def plt_img(img:torch.Tensor,save_path:str):
    """
        plot the image and save it to save_path
    """
    assert len(img.shape)==3
    transform=transforms.Compose([
        transforms.TensorToPILImage(),])
    pil_img=transform(img.cpu())
    #use plt to plot the image without axis
    plt.axis('off')
    plt.imshow(pil_img)
    print('saving in'+save_path)
    plt.savefig(save_path,bbox_inches='tight',pad_inches=0)
def plt_imgs(imgs:torch.Tensor,save_path:str):
    """
        plot the image and save it to save_path
    """
    if len(imgs.shape)==3:
        plt_img(imgs,save_path+'plt.svg')
    assert len(imgs.shape)==4
    transform=transforms.Compose([
        transforms.ToPILImage(),])
    for i,img in enumerate(imgs):
        pil_img=transform(img.cpu())
        #use plt to plot the image without axis
        plt.axis('off')
        plt.imshow(pil_img)
        print('saving in'+save_path+f'plt_{i}.svg')
        plt.savefig(save_path+f'plt_{i}.svg',bbox_inches='tight',pad_inches=0)
        plt.savefig(save_path+f'plt_{i}.png',bbox_inches='tight',pad_inches=0)
def plt_difs(imgs:torch.Tensor,noised:torch.Tensor,save_path:str):
    assert(imgs.shape==noised.shape)
    assert(len(imgs.shape)==4)
    #plot both imgs and noised in one figure
    transform=transforms.Compose([
        transforms.ToPILImage(),])
    for i,(img,noise) in enumerate(zip(imgs,noised)):
        pil_img=transform(img.cpu())
        pil_noise=transform(noise.cpu())
        #use plt to plot the image without axis
        
        #plt pil_img and pil_noise in two subfigure
        plt.subplot(1,2,1)
        plt.imshow(pil_img)
        plt.subplot(1,2,2)
        plt.imshow(pil_noise)
        plt.axis('off')
        print('saving in'+save_path+f'plt_{i}.svg')
        plt.savefig(save_path+f'plt_{i}.svg',bbox_inches='tight',pad_inches=0)
        plt.savefig(save_path+f'plt_{i}.png',bbox_inches='tight',pad_inches=0)
def plt_noise(imgs:torch.Tensor,noised:torch.Tensor,save_path:str):
    assert(imgs.shape==noised.shape)
    assert(len(imgs.shape)==4)
    noise=noised-imgs
    noise+=noise.min() if noise.min()<0 else 0
    print('saving in '+save_path)
    plt_imgs(noise,save_path,bbox_inches='tight',pad_inches=0)
def plt_noises(imgs:torch.Tensor,noised:torch.Tensor,save_path:str):
    assert(imgs.shape==noised.shape)
    assert(len(imgs.shape)==4)
    noises=noised-imgs
    for i,noise in enumerate(noises):
        if noise.min()<0:
            sample_noise=noise-noise.min()
        sample_noise+=0.1
        #use PIL to plot the image 
        transform=transforms.Compose([
            transforms.ToPILImage(),])
        pil_noise=transform(sample_noise.cpu())
        #use plt to plot the image without axis
        plt.axis('off')
        plt.imshow(pil_noise)
        print('saving in'+save_path+f'plt_{i}.svg')
        plt.savefig(save_path+f'plt_{i}.svg',bbox_inches='tight',pad_inches=0)
        plt.savefig(save_path+f'plt_{i}.png',bbox_inches='tight',pad_inches=0)
