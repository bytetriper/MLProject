
import datasets
from torchvision import transforms
import torch
from transformers import CLIPProcessor,CLIPModel,CLIPImageProcessor,CLIPTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from PIL import Image
from typing import List
from Params import Params
from torch.utils.data import DataLoader
def test_map():
    Dats=datasets.load_dataset('nlphuji/flickr30k')['test'].train_test_split(test_size=0.02)['test']
    #remove the column 'sentids', 'split', 'img_id', 'filename' of the dataset
    Dats=Dats.remove_columns(['sentids','img_id','split','filename'])
    #map the Dats into 'img','txt'
    Transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.PILToTensor(),
        #transforms.ConvertImageDtype(torch.float32),
        #transforms.ToTensor()
    ])
    tokenizer=CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch16')
    def transform(x:List[Image.Image]):
        return [Transform(i)/255. for i in x]
    def cut_txt(x:List[List[str]]):
        return x.split('\n')[0]
    Dats=Dats.map(lambda x: {'image':transform(x['image']),'caption':x['caption']},batch_size=32,batched=True,num_proc=32)
    Dats.set_format('torch')
    def collate_fn(x:List[dict])->BatchEncoding:
        # x: [{img:tensor,caption:{input_ids: list[int],attention_mask:list[int]} },...]
        # return a BatchEncoding
        batched_data=tokenizer([i['caption'][0] for i in x],padding=True,return_tensors='pt')
        batched_data['image']=torch.stack([i['image'] for i in x])
        return batched_data
    loader=DataLoader(Dats,batch_size=32,num_workers=1,collate_fn=collate_fn,pin_memory_device="cuda:0",pin_memory=True)
    first_batch=next(iter(loader))
    print(first_batch)
def make_dataset():
    Dats=datasets.load_dataset('nlphuji/flickr30k',keep_in_memory=True)['test']
    #remove the column 'sentids', 'split', 'img_id', 'filename' of the dataset
    Dats=Dats.remove_columns(['sentids','img_id','split','filename'])
    #map the Dats into 'img','txt'
    Transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.PILToTensor(),
        #transforms.ConvertImageDtype(torch.float32),
        #transforms.ToTensor()
    ])
    print(Dats)
    tokenizer=CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch16')
    #set num proc to maximum
    def transform(x:List[Image.Image]):
        return [Transform(i) for i in x]
    def transform_txt(x:List[List[str]]):
        return tokenizer(x[:][0],return_tensors='pt',padding=True)
    Dats=Dats.map(lambda x: {'image':transform(x['image'])},batched=True,batch_size=64,num_proc=5)
    #Dats=Dats.map(lambda x: {'img':Transform(x['image']),'txt':x['caption']})
    #split the dataset into train and test,val with ratio: 8:1:1
    Dats=Dats.train_test_split(test_size=0.1)
    #get the train dataset
    train_dataset=Dats['train']
    #train_dataset.set_format('torch')
    #split the train dataset into train and val with ratio: 9:1
    train_dataset=train_dataset.train_test_split(test_size=0.1)
    print('train_dataset:',train_dataset)
    #save the train dataset
    train_dataset.save_to_disk('/root/autodl-tmp/fool_clip/train_dataset')
    #save the test dataset
    test_dataset=Dats['test']
    #test_dataset.set_format('torch')
    print('test_dataset:',test_dataset)
    test_dataset.save_to_disk('/root/autodl-tmp/fool_clip/test_dataset')
def get_tiny_imagenet_dataset():
    dataset = datasets.load_dataset("Maysee/tiny-imagenet",keep_in_memory=True)
    purified=dataset.map(
        lambda x: {'image':x['image'].convert('RGB'),'label':x['label']},
        num_proc=15
    )
    purified.save_to_disk('/root/autodl-tmp/fool_clip/tiny-imagenet')
def get_cifar10_dataset():
    dataset=datasets.load_dataset('cifar10',keep_in_memory=True)
    purified=dataset.map(
        lambda x: {'image':x['img'].convert('RGB'),'label':x['label']},
        num_proc=15
    )
    print(purified)
    purified.save_to_disk('/root/autodl-tmp/fool_clip/cifar10')
def get_cifar100_dataset():
    dataset=datasets.load_dataset('cifar100',keep_in_memory=True)
    print(dataset)
    purified=dataset.map(
        lambda x: {'image':x['img'].convert('RGB'),'label':x['fine_label']},
        num_proc=15
    )
    purified.save_to_disk('/root/autodl-tmp/fool_clip/cifar100')
def get_imagenet_dataset():
    dataset = datasets.load_dataset("mrm8488/ImageNet1K-val",cache_dir='../Imagenet')
    dataset.save_to_disk('/root/autodl-tmp/datasets/ImageNet1K-val')
    print(dataset)
def split_dataset():
    dataset=datasets.load_from_disk(Params['train_dataset_path'])['train']
    dataset=dataset.filter(lambda x: x['label'] is not None and x['label']!=-1 and x['label']>=0 and x['label']<=1000 , num_proc=15)
    print(dataset)
    dataset=dataset.train_test_split(test_size=0.02)['test']
    print(dataset)
    dataset.save_to_disk('/root/autodl-tmp/datasets/ImageNet1k-split')
if __name__ == "__main__":
    #get_cifar100_dataset()
    #make_dataset()
    #test_map()
    #test_dict=[{'a':'a'},{'a':'c'}]
    #print(test_dict[0]['a'])
    #dataset=datasets.load_dataset('imagenet')
    #get_imagenet_dataset()
    #split_dataset()
    get_cifar10_dataset()