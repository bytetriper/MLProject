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
from transformers.tokenization_utils_base import BatchEncoding
import multiprocessing as mp
Params = {
    'lr': 1e-5,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_generator': True,
    'load_generator': False,
    'generator_path': 'generator.pth',
    'model_name': 'openai/clip-vit-base-patch16',
    'batch_size': 64,
    'epochs':1,
    'select_col': 0,  # should be in range [1-5]
    'num_workers': 4,
    "noise_bound": 0.1,

}


class Fool_CLip(nn.Module):
    def __init__(self, bound: float = Params['noise_bound']):
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
        self.bounds = bound
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # use the generator to generate the noise of the image
        noise = self.generator(x)
        # clip the noise to the range of [0,bound]
        noise = torch.clip(noise, -self.bounds, self.bounds)
        # add the noise to the image
        x = torch.clip(x + noise, 0, 1)
        return x


class Wrapper(nn.Module):
    def __init__(self, half: bool = False):
        super(Wrapper, self).__init__()
        self.gen = Fool_CLip()
        self.gen2clip = Gen_To_Clip_Processor()
        self.target = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPTokenizerFast.from_pretrained(
            Params['model_name'])
        if Params['load_generator']:
            self.gen.load_state_dict(torch.load(Params['generator_path']))
        if half:
            self.gen.half()
            self.target.half()
        self.half = half
        self.gen.to(Params['device'])
        self.gen.eval()
        self.target.to(Params['device'])
        self.target.eval()
        self.optim = optim.Adam([{'params':self.gen.parameters(),
           'initial_lr':1 }], lr=Params['lr'])
        def warm_up(epoch):
            #before a hundred epoch, the lr satisfy lr=epoch*1e-5
            if epoch<400:
                return (epoch//4)*1e-5
            #otherwise, the lr satisfy a weight dacay of 0.85 per a hundred epoch
            else:
                return 2e-4
        self.schel = optim.lr_scheduler.LambdaLR(self.optim, warm_up, last_epoch=0)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        # noise the image
        img = inputs["pixel_values"]
        noised = self.gen(img)
        # noised = img
        # preprocess the image
        noised = self.gen2clip(noised)
        inputs["pixel_values"] = noised
        # get the logits
        logits = self.target(**inputs).logits_per_image
        return logits
    def __call__(self, inputs: BatchEncoding) -> torch.Tensor:
        return self.forward(inputs)
    def loss(self, logits):
        # logits : [img_size,txt_size]
        # label  : [img_size]
        # label[batch][k]=k
        label = torch.arange(logits.shape[0]).to(Params['device'])
        loss = nn.CrossEntropyLoss()(logits, label)
        return loss
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
        for epoch in range(Params['epochs']):
            tbar = tqdm(range(len(dataloader)))
            tbar.set_description_str(f"epoch:{epoch}/{Params['epochs']}")
            avg_loss = 0
            total_loss = 0
            for i, data in enumerate(dataloader):
                if self.half:
                    data['pixel_values'] = data['pixel_values'].half()
                data.to(Params['device'], non_blocking=True)
                logits = self.forward(data)
                loss = -self.loss(logits)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += -loss.item()
                avg_loss = total_loss/(i+1)
                tbar.set_postfix_str(
                    f"loss:{avg_loss:5e},lr:{self.optim.param_groups[0]['lr']:4e}")
                tbar.update(1)
                loss_history.append(avg_loss)
                self.schel.step()
            if Params['save_generator']:
                torch.save(self.gen.state_dict(), Params['generator_path'])
        return loss_history


def train_model(model: Wrapper, dataset: datasets.DatasetDict):
    train_set = dataset['train']
    # remove 'image' 'caption' from the dataset
    train_set.set_format('torch', columns=['image', 'caption'])
    tokenizer = CLIPTokenizerFast.from_pretrained(
        Params['model_name'])

    def collate_fn(x: list[dict]) -> BatchEncoding:
        # x: [{img:tensor,caption:{input_ids: list[int],attention_mask:list[int]} },...]
        # return a BatchEncoding
        batched_data = tokenizer([i['caption'][Params['select_col']][:100]
                                 for i in x], padding=True, return_tensors='pt')
        batched_data['pixel_values'] = torch.stack(
            [i['image'].float()/255 for i in x])
        return batched_data
    train_loader = DataLoader(train_set, batch_size=Params['batch_size'], num_workers=Params['num_workers'], shuffle=True,
                              collate_fn=collate_fn, pin_memory_device="cuda:0", pin_memory=True)
    loss_history = model.train(train_loader)
    plt.plot(loss_history)
    plt.savefig('loss.png')


def test_correctness():
    train_set = datasets.load_from_disk(
        '/root/autodl-tmp/fool_clip/train_dataset', keep_in_memory=True)['train']
    train_set.set_format('torch', columns=['image', 'caption'])
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(Params['device'])
    tokenizer = CLIPTokenizerFast.from_pretrained(
        Params['model_name'])

    def collate_fn(x: list[dict]) -> BatchEncoding:
        # x: [{img:tensor,caption:{input_ids: list[int],attention_mask:list[int]} },...]
        # return a BatchEncoding
        batched_data = tokenizer([i['caption'][0][:100]
                                 for i in x], padding=True, return_tensors='pt')
        batched_data['pixel_values'] = torch.stack(
            [i['image'].float()/255 for i in x])
        return batched_data
    loader = DataLoader(
        train_set, batch_size=Params['batch_size'], shuffle=False, collate_fn=collate_fn, pin_memory=True, pin_memory_device='cuda:0')
    cnt = 0
    label = torch.arange(Params['batch_size']).to(Params['device'])
    imgp= Gen_To_Clip_Processor()
    for data in loader:
        inputs = data.to(Params['device'])
        inputs['pixel_values'] = imgp(inputs['pixel_values'])
        output = model(inputs)
        print(output.logits_per_image.shape)
        prob=torch.softmax(output.logits_per_image,dim=1)
        choice=torch.argmax(prob,dim=1)
        print(choice)
        cnt+=1
        if cnt==5:
            break


def test_txt():
    train_set = datasets.load_from_disk(
        '/root/autodl-tmp/fool_clip/train_dataset', keep_in_memory=True)['train']
    train_set.set_format('torch', columns=['image', 'caption'])
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch16").to(Params['device'])
    model.eval()
    tokenizer = CLIPTokenizerFast.from_pretrained(
        Params['model_name'])

    def collate_fn(x: list[dict]) -> BatchEncoding:
        # x: [{img:tensor,caption:{input_ids: list[int],attention_mask:list[int]} },...]
        # return a BatchEncoding
        batched_data = tokenizer([i['caption'][0][:100]
                                 for i in x], padding=True, return_tensors='pt')
        batched_data['pixel_values'] = torch.stack(
            [i['image'].float()/255 for i in x])
        return batched_data
    loader = DataLoader(train_set, batch_size=32, num_workers=15,
                        collate_fn=collate_fn, pin_memory_device="cuda:0", pin_memory=True)
    tbar = tqdm(range(len(loader)))
    for data in loader:
        inputs = data.to(Params['device'], non_blocking=True)
        output = model(**inputs)
        tbar.update(1)

def test_zero_shot_classfication(model:Wrapper=None):
    train_set = datasets.load_from_disk(
        '/root/autodl-tmp/fool_clip/train_dataset', keep_in_memory=True)['test']
    train_set.set_format('torch', columns=['image', 'caption'])
    tokenizer = CLIPTokenizerFast.from_pretrained(
        Params['model_name'])
    standard_mode=False
    if model is None:
        model=CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(Params['device'])
        standard_mode=True
    def collate_fn(x: list[dict]) -> BatchEncoding:
        # x: [{img:tensor,caption:{input_ids: list[int],attention_mask:list[int]} },...]
        # return a BatchEncoding
        batched_data = tokenizer([i['caption'][Params['select_col']][:100]
                                 for i in x], padding=True, return_tensors='pt')
        batched_data['pixel_values'] = torch.stack(
            [i['image'].float()/255 for i in x])
        return batched_data
    train_loader = DataLoader(train_set, batch_size=Params['batch_size'], num_workers=Params['num_workers'], shuffle=True,
                              collate_fn=collate_fn, pin_memory_device="cuda:0", pin_memory=True)
    tbar = tqdm(range(len(train_loader)))
    total=0
    correct=0
    for data in train_loader:
        inputs = data.to(Params['device'], non_blocking=True)
        if not standard_mode:
            output = model(inputs)
            prob = output
        else:
            output= model(**inputs)
            prob=torch.softmax(output.logits_per_image,dim=1)
        choice=torch.argmax(prob,dim=1)
        #update correct and total
        total+=choice.shape[0]
        correct+=(choice==torch.arange(choice.shape[0],device=Params['device'])).sum().item()
        tbar.update(1)
    #return accuracy
    #print a pretty message to stdout
    print(f"Accuracy: {correct/total}")
    return correct/total

def visualization(model:Wrapper,img:torch.Tensor):
    #return a noised image in PIL form
    #img: a tensor of shape (3,224,224)
    
    #add the noise
    img= img.to(Params['device']).float().unsqueeze(0) /255.
    noised=model.gen(img)
    print(noised.max(),noised.min(),noised.mean())
    plt.imshow(noised.squeeze(0).detach().cpu().permute(1,2,0))
    plt.savefig('noised-plt.png')
    #convert to PIL
    noised=transforms.ToPILImage()(noised.squeeze(0).cpu())
    return noised
def get_visualization():
    model=Wrapper()
    #acc=test_zero_shot_classfication(model=model)
    dataset=datasets.load_from_disk('/root/autodl-tmp/fool_clip/train_dataset',keep_in_memory=True)['test']
    dataset.set_format('torch',columns=['image','caption'])
    source_img=dataset[176]['image']
    source_pil=transforms.ToPILImage()(source_img.float()/255.)
    source_pil.save('source.png')
    img=visualization(model,source_img)
    img.save('noised.png')
def compare_noise():
    model=Wrapper()
    #acc=test_zero_shot_classfication(model=model)
    dataset=datasets.load_from_disk('/root/autodl-tmp/fool_clip/train_dataset',keep_in_memory=True)['test']
    dataset.set_format('torch',columns=['image','caption'])
    id1=176
    id2=250
    source1=dataset[id1]['image'].to(Params['device']).float().unsqueeze(0)/255.
    source2=dataset[id2]['image'].to(Params['device']).float().unsqueeze(0)/255.
    noised1=model.gen(source1)-source1
    noised2=model.gen(source2)-source2
    #print the difference
    print(torch.abs(noised1-noised2).mean())
def get_noise():
    model=Wrapper()
    dataset=datasets.load_from_disk('/root/autodl-tmp/fool_clip/train_dataset',keep_in_memory=True)['test']
    dataset.set_format('torch',columns=['image','caption'])
    source1=dataset[133]['image'].to(Params['device']).float().unsqueeze(0)/255.
    noise=model.gen.generator(source1)
    noise=noise.clip(-Params['noise_bound'],Params['noise_bound'])+Params['noise_bound']
    print(noise.max(),noise.min(),noise.mean())
    plt.imshow(noise.squeeze(0).detach().cpu().permute(1,2,0))
    plt.savefig('noise.png')
    # save the noise in numpy form
    np.save('noise.npy',noise.squeeze(0).detach().cpu().permute(1,2,0).numpy())
def test_noise():
    #load noise from noise.npy
    noise=np.load('noise.npy')
    #transfrom noise to tensor
    noise=torch.from_numpy(noise).permute(2,0,1)-Params['noise_bound']
    #load model
    model=CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(Params['device'])

    dataset=datasets.load_from_disk('/root/autodl-tmp/fool_clip/train_dataset',keep_in_memory=True)['test']
    dataset.set_format('torch',columns=['image','caption'])
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
    def collate_fn(x: list[dict]) -> BatchEncoding:
        # x: [{img:tensor,caption:{input_ids: list[int],attention_mask:list[int]} },...]
        # return a BatchEncoding
        batched_data = tokenizer([i['caption'][Params['select_col']][:100]
                                 for i in x], padding=True, return_tensors='pt')
        batched_data['pixel_values'] = torch.stack(
            [(i['image'].float()/255 + noise).clip(0,1) for i in x])
        return batched_data
    train_loader = DataLoader(dataset, batch_size=32, num_workers=Params['num_workers'], shuffle=True,
                              collate_fn=collate_fn, pin_memory_device="cuda:0", pin_memory=True)
    tbar = tqdm(range(len(train_loader)))
    total=0
    correct=0
    for data in train_loader:
        inputs = data.to(Params['device'], non_blocking=True)
        output= model(**inputs)
        prob=torch.softmax(output.logits_per_image,dim=1)
        choice=torch.argmax(prob,dim=1)
        #update correct and total
        total+=choice.shape[0]
        correct+=(choice==torch.arange(choice.shape[0],device=Params['device'])).sum().item()
        tbar.update(1)
    #return accuracy
    #print a pretty message to stdout
    print(f"Accuracy: {correct/total}")
    return correct/total
if __name__ == "__main__":
    #test_correctness()
    # test_txt()
    train_set = datasets.load_from_disk(
        '/root/autodl-tmp/fool_clip/train_dataset' , keep_in_memory=True)
    model = Wrapper()
    train_model(model, train_set)
    test_zero_shot_classfication(model)
    #get_visualization()
    #compare_noise()
    #get_noise()
    #test_noise()