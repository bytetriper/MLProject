from PIL import Image
import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, trainer, CLIPImageProcessor, CLIPTokenizerFast
from torchvision.models.resnet import BasicBlock, Bottleneck
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
    'batch_size': 128,
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
        self.generator = nn.Sequential(
            # input shape: [batch_size,3,224,224]
            # do down_sampling
            # output every layer's shape in note
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size,64,224,224]
            nn.Conv2d(64, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size,64,112,112]
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [batch_size,128,56,56]
            nn.Conv2d(128, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [batch_size,128,28,28]
            # now do some residual blocks
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            # [batch_size,128,28,28]
            # do up_sampling
            nn.ConvTranspose2d(128, 128, 5, 2, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [batch_size,128,56,56]
            nn.ConvTranspose2d(128, 64, 5, 2, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size,64,112,112]
            nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size,64,224,224]
            # do a conv to get the output
            nn.Conv2d(64, 3, 7, 1, 3),
        )
        self.bounds = bound

    def forward(self, x):
        # use the generator to generate the noise of the image
        noise = self.generator(x)
        # clip the noise to the range of [0,bound]
        noise = torch.clip(noise, 0, self.bounds)
        # add the noise to the image
        x = torch.clip(x + noise, 0, 1)
        return x


class Wrapper(nn.Module):
    def __init__(self):
        super(Wrapper, self).__init__()
        self.gen = Fool_CLip()
        self.gen2clip = Gen_To_Clip_Processor()
        self.target = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPTokenizerFast.from_pretrained(
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
        #noised = img
        # preprocess the image
        inputs = self.processor(txt, return_tensors="pt",
                                padding=True).to(Params['device'])
        noised = self.gen2clip(noised)
        inputs["pixel_values"] = noised
        # get the logits
        logits = self.target(**inputs).logits_per_image
        return logits

    def loss(self, logits):
        # logits : [img_size,txt_size]
        # label  : [img_size]
        # label[batch][k]=k
        label = torch.arange(logits.shape[0]).to(Params['device'])
        loss = nn.CrossEntropyLoss()(logits, label)
        return loss

    def process_patch(self, patch: torch.Tensor):
        return patch.float()/255.

    def train(self, dataloader: DataLoader, select_col: int = 0) -> list[float]:
        self.gen.train()
        self.target.train()
        loss_history = []
        for epoch in range(Params['epochs']):
            tbar = tqdm(range(len(dataloader)))
            tbar.set_description_str(f"epoch:{epoch}/{Params['epochs']}")
            avg_loss = 0
            total_loss = 0
            for i, data in enumerate(dataloader):
                img = self.process_patch(data['image'].to(Params['device']))
                txt = data['caption'][select_col]
                logits = self.forward(img, txt)
                loss = -self.loss(logits)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += -loss.item()
                avg_loss = total_loss/(i+1)
                tbar.set_postfix_str(
                    f"loss:{avg_loss:5e},lr:{self.optim.param_groups[0]['lr']:4e}")
                tbar.update(1)
                loss_history.append(-loss.item())
            self.schel.step()
        if Params['save_generator']:
            torch.save(self.gen.state_dict(), Params['generator_path'])
        return loss_history


def train_model(model: Wrapper, dataset: datasets.DatasetDict):
    train_set = dataset['train']
    # remove 'image' 'caption' from the dataset
    train_set.set_format('torch', columns=['image', 'caption'])

    train_loader = DataLoader(
        train_set, batch_size=Params['batch_size'], shuffle=True)
    loss_history = model.train(train_loader)
    plt.plot(loss_history)
    plt.savefig('loss.png')


def test_correctness():
    train_set = datasets.load_from_disk(
        '/root/autodl-tmp/fool_clip/train_dataset', keep_in_memory=True)['train']
    train_set.set_format('torch', columns=['image', 'caption'])
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained(
        Params['model_name'])
    loader = DataLoader(
        train_set, batch_size=Params['batch_size'], shuffle=False)
    cnt = 0
    for data in loader:
        inputs = processor(
            text=data['caption'][0], images=data['image'], return_tensors='pt', padding=True)
        output = model(**inputs)
        print(output.logits_per_image.shape)
        loss = nn.CrossEntropyLoss()(output.logits_per_image,
                                     torch.arange(output.logits_per_image.shape[0]))
        print(loss)
        cnt += 1
        if cnt == 10:
            break
def test_txt():
    train_set = datasets.load_from_disk(
        '/root/autodl-tmp/fool_clip/train_dataset', keep_in_memory=True)['train']
    train_set.set_format('torch', columns=['image', 'caption'])
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(Params['device'])
    model.eval()
    processor = CLIPTokenizerFast.from_pretrained(
        Params['model_name'])
    loader = DataLoader(
        train_set, batch_size=Params['batch_size'], shuffle=False)
    tbar=tqdm(range(len(loader)))
    for data in loader:
        inputs = processor(
            text=data['caption'][0], return_tensors='pt', padding=True).to(Params['device'])
        img=data['image'].float().to(Params['device'])
        inputs['pixel_values']=img
        output = model(**inputs)
        tbar.update(1)
if __name__ == "__main__":
    #test_correctness()
    test_txt()
    #train_set = datasets.load_from_disk(
    #   '/root/autodl-tmp/fool_clip/train_dataset',keep_in_memory=True)
    #model=Wrapper()
    #train_model(model,train_set)
