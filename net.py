import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizerFast
from transformers.models.clip.modeling_clip import CLIPOutput
from torchvision.models.resnet import BasicBlock
import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import *
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
from typing import Union, List, Tuple
from Params import Params
from torch.utils.tensorboard import SummaryWriter

"""
old setting:
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
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
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
"""
class Fool_CLip(nn.Module):
    def __init__(self, bound: float = Params['noise_bound']):
        super(Fool_CLip, self).__init__()
        # train a gen to fool clip from scratch
        self.generator = nn.Sequential(
            # input shape: [batch_size,3,224,224]
            # do down_sampling
            # output every layer's shape in note
            nn.Conv2d(3, 32, 9, 1, 4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [batch_size,64,224,224]
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size,64,112,112]
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [batch_size,128,56,56]
            # now do some residual blocks
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            # do up_sampling
            # [batch_size,128,56,56]
            nn.ConvTranspose2d(128, 64, 5, 2, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size,64,112,112]
            nn.ConvTranspose2d(64, 32, 5, 2, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [batch_size,64,224,224]
            # do a conv to get the output
            nn.Conv2d(32, 3, 9, 1, 4),
        )
        self.bounds = bound
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # use the generator to generate the noise of the image
        noise = self.generator(x)
        # clip the noise to the range of [-bound,+bound]
        noise = torch.clip(noise, -self.bounds, self.bounds)
        # add the noise to the image
        returned = torch.clip(x + noise, 0, 1)
        return returned, returned-x


class Wrapper(nn.Module):
    def __init__(self, half: bool = False):
        super(Wrapper, self).__init__()
        self.gen = Fool_CLip()
        self.gen2clip = Gen_To_Clip_Processor()
        self.target = CLIPModel.from_pretrained(
            Params['model_name'], local_files_only=True)
        self.processor = CLIPTokenizerFast.from_pretrained(
            Params['model_name'], local_files_only=True)
        if Params['load_generator']:
            print('load generator from {}'.format(
                Params['generator_path']))
            self.gen.load_state_dict(torch.load(Params['generator_path']))
        if half:
            self.gen.half()
            self.target.half()
        self.half = half
        self.gen.to(Params['device'])
        self.target.to(Params['device'])
        self.eval()
        self.optim = optim.Adam([{'params': self.gen.parameters(),
                                  'initial_lr': 1}], lr=Params['lr'])
        self.cosloss_label=torch.ones(Params['batch_size'],dtype=torch.int,device=Params['device'])
        def warm_up(epoch):
            # before a hundred epoch, the lr satisfy lr=epoch*1e-5
            if epoch < 400:
                return (epoch//4)*1e-5
            # otherwise, the lr satisfy a weight dacay of 0.85 per a hundred epoch
            else:
                return 2e-4 * 0.85 ** ((epoch-400)//100)
        self.schel = optim.lr_scheduler.LambdaLR(
            self.optim, warm_up, last_epoch=1)

    def forward(self, inputs: BatchEncoding) -> Tuple[CLIPOutput, torch.Tensor]:
        if self.training:
            original_outputs = self.target(**inputs, return_loss=True)
            self.original_outputs = original_outputs
        # noise the image
        img = inputs["pixel_values"]
        noised, noise = self.gen(img)
        # noised = img
        # preprocess the image
        noised = self.gen2clip(noised)
        inputs["pixel_values"] = noised
        # get the logits
        outputs = self.target(**inputs, return_loss=True)
        return outputs, noise

    def __call__(self, inputs: BatchEncoding) -> torch.Tensor:
        return self.forward(inputs)

    def loss(self, outputs: CLIPOutput):
        coslossfunc = nn.CosineEmbeddingLoss()
        #compute the clip cos loss for image_embeds
        img_img_loss= coslossfunc(outputs.image_embeds, self.original_outputs.image_embeds,self.cosloss_label)
        #compute the clip cos loss for image-text_embedss
        img_txt_loss= coslossfunc(outputs.image_embeds,self.original_outputs.text_embeds,self.cosloss_label)
        return img_img_loss,img_txt_loss
    def eval(self):
        self.gen.eval()
        self.target.eval()

    def train(self):
        self.gen.train()
        self.target.train()

    def train(self, dataloader: DataLoader) -> list[float]:
        self.gen.train()
        self.target.train()
        loss_history = []
        writer = SummaryWriter(Params['summary_path'])
        for epoch in range(Params['epochs']):
            tbar = tqdm(range(len(dataloader)))
            tbar.set_description_str(f"epoch:{epoch}/{Params['epochs']}")
            avg_loss = 0
            total_loss = 0
            for i, data in enumerate(dataloader):
                if self.half:
                    data['pixel_values'] = data['pixel_values'].half()
                data.to(Params['device'], non_blocking=True)
                outputs, noise = self.forward(data)
                imgloss,txtloss = self.loss(outputs)
                # noiseloss,l1loss,stdloss = self.loss(noise)
                loss = -(5* imgloss+txtloss)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += -loss.item()
                avg_loss = total_loss/(i+1)
                writer.add_scalar('loss', loss.item(), epoch*len(dataloader)+i)
                writer.add_scalar(
                    'lr', self.optim.param_groups[0]['lr'], epoch*len(dataloader)+i)
                writer.add_scalar('loss', avg_loss, epoch*len(dataloader)+i)
                tbar.set_postfix_str(
                    f"loss:{avg_loss:.3e},lr:{self.optim.param_groups[0]['lr']:.3e},imgloss:{5*imgloss.item():.3e},txtloss:{txtloss.item():.3e}")
                # tbar.set_postfix_str(
                #    f"loss:{loss:.3e} closs:{cliploss:.3e},lr:{self.optim.param_groups[0]['lr']:.3e},l1loss:{l1loss:.3e},stdloss:{stdloss:.3e}")
                tbar.update(1)
                loss_history.append(avg_loss)
                self.schel.step()
            if Params['save_generator']:
                print('save generator to {}'.format(
                    Params['generator_path']))
                torch.save(self.gen.state_dict(), Params['generator_path'])
        return loss_history
