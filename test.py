from PIL import Image
import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, trainer, CLIPImageProcessor, CLIPTokenizerFast
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import PIL.Image as Image
from diy_processor import Img_Processor, Gen_To_Clip_Processor
from tqdm import tqdm
import torchvision.transforms as transforms
import multiprocessing as mp
Params = {
    'batch_size': 32,
    'lr': 0.001,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_generator': True,
    'load_generator': False,
    'generator_path': 'generator.pth',
    'model_name': 'openai/clip-vit-base-patch16',
    'batch_size': 32,
    'epochs': 1,

}


class Fool_CLip(nn.Module):
    def __init__(self, bound: float = 0.1):
        super(Fool_CLip, self).__init__()
        # train a gen to fool clip from scratch
        self.generator = resnet50(weights=None)
        self.bounds = bound

    def forward(self, x):
        # use the generator to generate the noise of the image
        noise = self.generator(x)
        # clip the noise to the range of [0,bound]
        noise = torch.clip(noise, 0, self.bounds)
        # add the noise to the image
        x = torch.clip(x + noise, 0, 1-self.bounds)
        return x


class Wrapper(nn.Module):
    def __init__(self):
        super(Wrapper, self).__init__()
        self.gen = Fool_CLip()
        self.gen2clip = Gen_To_Clip_Processor()
        self.target = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPImageProcessor.from_pretrained(
            Params['model_name'])
        if Params['load_generator']:
            self.gen.load_state_dict(torch.load(Params['generator_path']))
        self.gen.to(Params['device'])
        self.gen.eval()
        self.target.to(Params['device'])
        self.target.eval()
        self.optim = optim.Adam(self.gen.parameters(), lr=Params['lr'])
        self.schel = optim.lr_scheduler.StepLR(
            self.optim, step_size=1, gamma=0.1)

    def forward(self, img: torch.Tensor, txt: list[str]) -> torch.Tensor:
        # noise the image
        noised = self.gen(img)
        # preprocess the image
        inputs = self.processor(txt, return_tensors="pt",
                                padding=True).to(Params['device'])
        noised = self.gen2clip(noised)
        inputs["pixel_values"] = noised
        # get the logits
        logits = self.model(noised).logits_per_image
        return logits

    def loss(self, logits):
        #logits : [img_size,txt_size]
        #label  : [img_size]
        # label[batch][k]=k
        label = torch.arange(logits.shape[0]).to(Params['device'])
        loss = nn.CrossEntropyLoss()(logits, label)
        return loss

    def train(self, dataloader: DataLoader) -> list[float]:
        self.gen.train()
        self.target.train()
        loss_history = []
        for epoch in range(Params['epochs']):
            tbar = tqdm(range(len(dataloader)))
            tbar.set_description_str(f"epoch:{epoch}/{Params['epochs']}")
            avg_loss = 0
            total_loss = 0
            for i, (img, txt) in enumerate(dataloader):
                logits = self.forward(img, txt)
                loss = self.loss(logits)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                avg_loss = total_loss/(i+1)
                tbar.set_postfix_str(
                    f"loss:{avg_loss},lr:{self.optim.param_groups[0]['lr']}")
                tbar.update(1)
                loss_history.append(loss.item())
            self.schel.step()
        if Params['save_generator']:
            torch.save(self.gen.state_dict(), Params['generator_path'])
        return loss_history


def train_model(model: Wrapper, dataset: datasets.DatasetDict):
    train_set = dataset['train'].train_test_split(train_size=0.003)['train']
    # remove 'image' 'caption' from the dataset
    Transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    train_set.set_format('torch', columns=['image', 'caption'])
    
    train_loader = DataLoader(
        train_set, batch_size=Params['batch_size'], shuffle=True)
    for s in train_loader:
        print(s)
        break


if __name__ == "__main__":
    train_set = datasets.load_from_disk(
        '/root/autodl-tmp/fool_clip/train_dataset')
    model = Wrapper()
    train_model(model, train_set)
