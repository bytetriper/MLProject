import numpy as np
import PIL.Image as Image
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    make_list_of_images)
from typing import Union
import torch


class Img_Processor():
    def __init__(self, img_size: int = 224, Rescale_Factor: int = 255, device: torch.device = torch.device('cpu')):
        self.img_size = img_size
        self.Rescale_Factor = Rescale_Factor
        self.device = device
    def __call__(self, img: Union[Image.Image, list[Image.Image]], do_rescale: bool = True, convert_rgb: bool = True) -> torch.Tensor:
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
