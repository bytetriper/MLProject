import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection
from transformers.models.clip.modeling_clip import CLIPVisionModelOutput
from torchvision.models.resnet import BasicBlock
import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import *
from tqdm import tqdm
from typing import Union, List, Tuple
from Params import Params
from torch.utils.tensorboard import SummaryWriter
from diffusers import UNet2DModel
from apex import amp
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from diffusers import AutoencoderKL
import torch.functional as F
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
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
        self.BatchNorm_PreProcess = nn.BatchNorm2d(3)
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

    def forward(self, x, preprocess: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if preprocess:
            preprocessed = self.BatchNorm_PreProcess(x)
        else:
            preprocessed = x
        # use the generator to generate the noise of the image
        noise = self.generator(preprocessed)
        # clip the noise to the range of [-bound,+bound]
        noise = torch.clip(noise, -self.bounds, self.bounds)
        # add the noise to the image
        returned = torch.clip(preprocessed + noise, 0, 1)
        return returned, returned-x


class U_Fool(nn.Module):
    def __init__(self, bound: float = Params['noise_bound'], step: int = Params['step']):
        super(U_Fool, self).__init__()
        # use a unet 2d to generate the noise
        self.down_sample = nn.Sequential(
            # input shape: [batch_size,3,224,224]
            # do down_sampling
            # output every layer's shape in note
            nn.Conv2d(3, 16, 9, 1, 4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # [batch_size,64,224,224]
            nn.Conv2d(16, 16, 5, 2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # [batch_size,64,112,112]
            nn.Conv2d(16, 16, 5, 2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # [batch_size,128,56,56]
        )
        self.Base_Unet = UNet2DModel(
            sample_size=56,  # the target image resolution
            in_channels=16,  # the number of input channels, 3 for RGB images
            out_channels=16,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            # the number of output channels for each UNet block
            block_out_channels=(128, 256, 256, 512),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.up_sample = nn.Sequential(
            # do up_sampling
            # [batch_size,16,56,56]
            nn.ConvTranspose2d(16, 16, 5, 2, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # [batch_size,16,112,112]
            nn.ConvTranspose2d(16, 16, 5, 2, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # [batch_size,16,224,224]
            # do a conv to get the output
            nn.Conv2d(16, 3, 9, 1, 4),
        )

        # self.Base_Unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        self.bounds = bound/max(step, 1)
        self.step = step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x

        for step in range(self.step):
            subx = self.down_sample(x)
            noise = self.Base_Unet(subx, step)
            noise = self.up_sample(noise.sample)
            noise = torch.clip(noise, -self.bounds, self.bounds)
            x = torch.clip(x+noise, 0, 1)
        return x, x-original

    def enable_gradient_checkpointing(self):
        """
        deprecated
        """
        self.Base_Unet.enable_gradient_checkpointing()

class Integrated_Model(nn.Module):
    def __init__(self, mode: str = 'base', naive_target: str = 'clip'):
        super(Integrated_Model, self).__init__()
        if mode == 'base':
            print('use base fool clip')
            self.gen = Fool_CLip()
        elif mode == 'unet':
            print('use unet fool clip')
            self.gen = U_Fool()
            # self.gen.enable_gradient_checkpointing()
        else:
            raise ValueError('mode must be base or unet')
        if naive_target == 'clip':
            self.gen2clip = Gen_To_Clip_Processor()
            self.target = CLIPVisionModelWithProjection.from_pretrained(
                Params['model_name'], local_files_only=True)
            self.target._set_gradient_checkpointing(
                self.target.vision_model, True)
        elif naive_target == 'resnet50':
            self.target = models.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError('naive_target must be clip or resnet50')
        self.naive_target = naive_target
    def forward(self, x: torch.Tensor) -> CLIPVisionModelOutput:
        img = x
        if self.naive_target == 'clip':
            if self.training:
                self.original_outputs = self.target(
                    x, output_hidden_states=True)
            # print('gen')
            noised, _ = self.gen(img)
            fnoised = self.gen2clip(noised)
            outputs = self.target(fnoised, output_hidden_states=True)
            outputs['noised'] = noised
            return outputs
        elif self.naive_target == 'resnet50':
            if self.training:
                outputs, hidden_state = self.target(
                    x, output_hidden_states=True)
                self.original_outputs = CLIPVisionModelOutput(
                    image_embeds=None,
                    last_hidden_state=outputs,
                    hidden_states=hidden_state,
                    attentions=None,
                )
            noised, _ = self.gen(img)
            outputs, hidden_state = self.target(
                noised, output_hidden_states=True)
            Voutput = CLIPVisionModelOutput(
                image_embeds=None,
                last_hidden_state=outputs,
                hidden_states=hidden_state,
                attentions=None,
            )
            Voutput['noised'] = noised
            return Voutput

    def train(self):
        self.gen.train()
        self.target.train()

    def eval(self):
        self.gen.eval()
        self.target.eval()


class Integreted_VAE_Model(nn.Module):
    def __init__(self, mode: str, target: str, with_gen:bool=False,input_img:torch.Tensor=None):
        super(Integreted_VAE_Model, self).__init__()
        self.mode=mode
        self.target_mode=target
        self.VAE = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-base",subfolder="vae").to(Params['device'])
        self.scale = 0.18215  # constant scale set in VAE encoder and decoder
        self.with_gen=with_gen
        #self.VAE.encoder.requires_grad_(False)  # freeze VAE encoder
        if with_gen:
            if mode == 'base':
                print('use resnet to generate')
                self.gen = Fool_CLip()
            elif mode == 'unet':
                print('use unet to generate')
                self.gen = U_Fool()
            self.gen=self.gen.to(Params['device'])
        else:
            if input_img is None:
                raise ValueError('input_img must be provided when with_gen is false')
            self.input_img=self.encode(input_img).detach()
            self.input_img.requires_grad=True
        if target == 'clip':
            self.target = CLIPVisionModelWithProjection.from_pretrained(
                Params['model_name'], local_files_only=True)
        else:
            self.target = models.resnet50(
                pretrained=False)
            self.target.load_state_dict(torch.load("./models/test_resnet.pth"))
        self.target=self.target.to(Params['device'])
        if not with_gen:
            self.original_outputs = self.forward(input_img.detach())
            self.original_outputs['last_hidden_state']=self.original_outputs['last_hidden_state'].detach()
            self.original_outputs['hidden_states']=[x.detach() for x in self.original_outputs['hidden_states']]
        print('model init done')
    def eval(self):
        self.VAE.eval()
        self.target.eval()
        if self.with_gen:
            self.gen.eval()
    def train(self):
        self.VAE.train()
        self.target.train()
        if self.with_gen:
            self.gen.train()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = 2*x-1.0
        return self.scale*self.VAE.encode(x).latent_dist.sample()
    def TransferToClipOutput(self,inputs:torch.Tensor)->CLIPVisionModelOutput:
        if self.target_mode == 'clip':
            outputs = self.target(inputs, output_hidden_states=True)
        elif self.target_mode == 'resnet50':
            outputs, hidden_state = self.target(
                inputs, output_hidden_states=True)
            outputs = CLIPVisionModelOutput(
                image_embeds=None,
                last_hidden_state=outputs,
                hidden_states=hidden_state,
                attentions=None,
            )
        else:
            raise ValueError('target must be clip or resnet50')
        return outputs
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = 1 / 0.18215 * latents
        image = self.VAE.decode(latents)['sample']
        image = (image/2 + 0.5).clamp(0, 1)
        return image
    def forward(self, x: torch.Tensor=None) -> CLIPVisionModelOutput:
        if self.with_gen:
            img = x
            if self.training:
                self.original_outputs = self.TransferToClipOutput(img)
            # print('gen')
            img=self.encode(img)
            noised, _ = self.gen(img)
            noised=self.decode(noised)
            if self.target_mode == 'clip':
                fnoised = self.gen2clip(noised)
            else:
                fnoised=noised    
            outputs = self.TransferToClipOutput(fnoised)
            return outputs
        else:
            img=self.decode(self.input_img)
            outputs = self.TransferToClipOutput(img)
            return outputs
        
class Wrapper():
    def __init__(self, mode: str = Params['base'], target: str = Params['target'], amp_mode: bool = Params['amp_mode'],image:torch.Tensor=None):
        super(Wrapper, self).__init__()
        #self.model = Integrated_Model(mode, target).to(Params['device'])
        self.model=Integreted_VAE_Model(mode,target,with_gen=False,input_img=image).to(Params['device'])
        if self.model.with_gen:
            self.optim = optim.AdamW([{'params': self.model.gen.parameters(),
                                      'initial_lr': 1}], lr=Params['lr'])
        else:
            self.optim = optim.AdamW([{'params': self.model.input_img,  
                                        'initial_lr': 1}], lr=Params['lr'])
        self.optim.zero_grad()
        self.cosloss_label = torch.ones(
            Params['batch_size'], dtype=torch.int, device=Params['device'])
        self.amp_mode = amp_mode
        self.target = target
        self.mode = mode
        if self.amp_mode:
            print('amp mode on')
            self.model, self.optim = amp.initialize(
                self.model, self.optim, opt_level=Params['opt_level'])
        if Params['load_generator']:
            print('load generator from {}'.format(
                Params['base_path'] if mode == 'base' else Params['unet_path']))
            self.load()
        C = 785
        #self.schel = optim.lr_scheduler.LambdaLR(
        #    self.optim, self.warm_up, last_epoch=1+Params['last_epoch']*C)
        self.schel=optim.lr_scheduler.StepLR(self.optim,step_size=20,gamma=1)
    def warm_up(self, epoch):
        # before a hundred epoch, the lr satisfy lr=epoch*1e-5
        if epoch < 200:
            return (epoch//5)*1e7
        # otherwise, the lr satisfy a weight dacay of 0.85 per a hundred epoch
        else:
            return 4e-4 * 0.9 ** ((epoch-200)//100)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.forward(inputs)

    def forward(self, inputs: torch.Tensor=None) -> torch.Tensor:
        return self.model.forward(inputs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def loss(self, outputs: CLIPVisionModelOutput):
        loss = 0
        if self.target == 'clip':
            coslossf = nn.CosineSimilarity(dim=1)
            [loss := loss+coslossf(outputs['hidden_states'][i], self.model.original_outputs['hidden_states'][i]).mean()
             for i in range(len(outputs['hidden_states']))]
            imgloss = coslossf(outputs['image_embeds'],
                               self.model.original_outputs['image_embeds']).mean()
        elif self.target == 'resnet50':
            coslossf = nn.CosineSimilarity(dim=1)
            loss = coslossf(outputs['last_hidden_state'],
                            self.model.original_outputs['last_hidden_state']).mean()
            imgloss = torch.tensor(0).cuda()
        return loss, imgloss
    def CEloss(self,outputs:CLIPVisionModelOutput,label:torch.Tensor):
        state=outputs['last_hidden_state']
        lossf=nn.CrossEntropyLoss()
        loss=lossf(state,label)
        return loss
    def load(self) -> None:
        path = Params['base_path'] if self.mode == 'base' else Params['unet_path']
        if self.amp_mode:
            print('load amp')
            checkpoint = torch.load(path)
            self.model.gen.load_state_dict(checkpoint['gen'])
            self.optim.load_state_dict(checkpoint['optim'])
            amp.load_state_dict(torch.load(path)['amp'])
        else:
            self.model.gen.load_state_dict(torch.load(path)['gen'])

    def save(self) -> None:
        print('mode:'+self.mode)
        path = Params['base_path'] if self.mode == 'base' else Params['unet_path']
        if self.amp_mode:
            print('save amp in {}'.format(path))
            torch.save({'gen': self.model.gen.state_dict(
            ), 'optim': self.optim.state_dict(), 'amp': amp.state_dict()}, path)
        else:
            print('save in {}'.format(path))
            torch.save({'gen': self.model.gen.state_dict()}, path)
    def gradient_ascent(self,epoch:int,label:torch.Tensor=None)->torch.Tensor:
        assert self.model.with_gen==False
        self.model.train()
        tbar=tqdm(range(epoch),desc='gradient ascent')
        data=self.model.input_img.clone()
        for t in tbar:
            self.optim.zero_grad()
            outputs=self.model()
            if label is None:
                loss,_=self.loss(outputs)
            else:
                loss=-self.CEloss(outputs,label)
            loss.backward()
            self.optim.step()
            self.schel.step()
            tbar.set_postfix({'loss':loss.item(),'lr':self.optim.param_groups[0]['lr']})
        print(torch.abs(data-self.model.input_img).mean())
        return self.model.decode(self.model.input_img)

    def train(self, dataloader: DataLoader, summary_folder: str = None) -> List[float]:
        self.model.train()
        loss_history = []
        if summary_folder is not None:
            writer = SummaryWriter(summary_folder)
        avg_loss = 0
        total_loss = 0
        for epoch in range(Params['epochs']):
            tbar = tqdm(range(len(dataloader)))
            tbar.set_description_str(f"epoch:{epoch}/{Params['epochs']}")
            for i, data in enumerate(dataloader):
                data = data.to(Params['device'], non_blocking=True)
                outputs = self.model(data)
                loss, imgloss = self.loss(outputs)
                # noiseloss,l1loss,stdloss = self.loss(noise)
                # loss = -loss
                self.optim.zero_grad()
                if self.amp_mode:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                        # loss = scaled_loss
                else:
                    loss.backward()
                self.optim.step()
                total_loss += loss.item()
                avg_loss = total_loss/(epoch*len(dataloader)+i+1)
                if summary_folder is not None:
                    writer.add_scalars(
                        'loss', {"loss": loss.item(), "avgloss": avg_loss}, (epoch+Params['last_epoch'])*len(dataloader)+i)
                    writer.add_scalar(
                        'lr', self.optim.param_groups[0]['lr'], (epoch+Params['last_epoch'])*len(dataloader)+i)
                    writer.add_scalars('loss', {'imgloss': imgloss.item(
                    )}, (epoch+Params['last_epoch'])*len(dataloader)+i)
                tbar.set_postfix_str(
                    f"loss:{avg_loss:.3e},lr:{self.optim.param_groups[0]['lr']:.3e},imgloss:{imgloss.item():.3e}")
                # tbar.set_postfix_str(
                #    f"loss:{loss:.3e} closs:{cliploss:.3e},lr:{self.optim.param_groups[0]['lr']:.3e},l1loss:{l1loss:.3e},stdloss:{stdloss:.3e}")
                tbar.update(1)
                loss_history.append(avg_loss)
                self.schel.step()
            if Params['save_generator']:
                self.save()
        return loss_history
