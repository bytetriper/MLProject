from net import Wrapper,Params
from transformers import CLIPTokenizerFast,CLIPModel,CLIPProcessor
import datasets
from torch.utils.data import DataLoader
from utils import *
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torchvision import transforms
import random
from time import time
from utils import *
import torch.nn as nn
from train import train_vae_model
def train_model(model: Wrapper, dataset: datasets.DatasetDict):
    train_set = dataset['train']
    # remove 'image' 'caption' from the dataset
    train_set.set_format('torch', columns=['image', 'caption'])
    train_loader = CLIPLoader(train_set, batch_size=Params['batch_size'], num_workers=Params['num_workers'], shuffle=True,
                              collate_fn=None, pin_memory_device="cuda:0", pin_memory=True)
    loss_history = model.train(train_loader)
    plt.plot(loss_history)
    plt.savefig('loss.png')


def test_correctness():
    train_set = datasets.load_from_disk(
        '/root/autodl-tmp/fool_clip/train_dataset', keep_in_memory=True)['train']
    train_set.set_format('torch', columns=['image', 'caption'])
    model = CLIPModel.from_pretrained(Params['model_path']).to(Params['device'])
    tokenizer = CLIPTokenizerFast.from_pretrained(
        Params['model_path'])

    def collate_fn(x: List[dict]) -> BatchEncoding:
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
    train_set = load_data_with_retry(
        Params["train_dataset_path"], keep_in_memory=True)['train']
    train_set.set_format('torch', columns=['image', 'caption'])
    model = CLIPModel.from_pretrained(
        Params["model_name"]).to(Params['device'])
    model.eval()

    loader = CLIPLoader(train_set, batch_size=32, num_workers=15,
                        collate_fn=None, pin_memory_device="cuda:0", pin_memory=True)
    tbar = tqdm(range(len(loader)))
    for data in loader:
        inputs = data.to(Params['device'], non_blocking=True)
        output = model(**inputs)
        tbar.update(1)

def test_zero_shot_classfication(model:Wrapper=None):
    print('testing zero shot classfication:using validation dataset with batch_size:',Params["batch_size"])
    train_set = load_data_with_retry(
        Params['train_dataset_path'], keep_in_memory=True)['test']
    train_set.set_format('torch', columns=['image', 'caption'])
    standard_mode=False
    if model is None:
        print('using standard model')
        model=CLIPModel.from_pretrained(Params['model_name'],local_files_only=True).to(Params['device'])
        standard_mode=True
    else:
        print('using fool model')
    train_loader = CLIPLoader(
        dataset=train_set, batch_size=Params['batch_size'], num_workers=Params['num_workers'], shuffle=True,
        collate_fn=None, pin_memory_device="cuda:0", pin_memory=True
    )
    tbar = tqdm(range(len(train_loader)))
    total=0
    correct=0
    model.eval()
    for data in train_loader:
        inputs = data.to(Params['device'], non_blocking=True)
        if not standard_mode:
            output,_ = model(inputs)
            prob = torch.softmax(output.logits_per_image,dim=1)
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
    print(f"\nAccuracy: {correct/total}\n")
    return correct/total

def visualization(model:Wrapper,img:torch.Tensor,noise_image_path:str=Params['noise_image_path'],noise_path:str=Params['noise_path']):
    #return a noised image in PIL form
    #img: a tensor of shape (3,224,224)
    
    #add the noise
    img= img.to(Params['device']).float().unsqueeze(0) /255.
    noised,noise=model.gen(img)
    #print the max and min, std and mean of the noise
    print(f"noise max:{noise.max()},noise min:{noise.min()},noise mean:{noise.mean()},noise std:{noise.std()}")
    #save the noise in npy form
    print('saving noise into '+noise_path)
    np.save(noise_path,noise.detach().cpu().numpy())
    #save the noise as a png
    print('saving noise image into '+noise_image_path)
    noise=transforms.ToPILImage()((noise+Params['noise_bound']).squeeze(0).cpu())
    noise.save(noise_image_path)
    #convert to PIL
    noised=transforms.ToPILImage()(noised.squeeze(0).cpu())
    return noised
def get_visualization(model:nn.Module=None,pid:int=-1,random_seed:int=-1,ori_img_path:str=Params['original_image_path'],noised_img_path:str=Params["noised_image_path"]):
    random.seed(time() if random_seed==-1 else random_seed)
    print('getting visualization')
    if model is None:
        model=Wrapper().model
    for para in model.gen.parameters():
        para.data=para.data.float()
    #acc=test_zero_shot_classfication(model=model)
    dataset=datasets.load_from_disk(Params['train_dataset_path'],keep_in_memory=True)['test']
    dataset.set_format('torch',columns=['image','caption'])
    source_img=dataset[random.randint(0,300) if pid==-1 else pid ]['image']
    source_pil=transforms.ToPILImage()(source_img.float()/255.)
    print('saving orginal image into '+ori_img_path)
    source_pil.save(ori_img_path)
    img=visualization(model,source_img)
    print('saving noised image into '+noised_img_path)
    img.save(noised_img_path)
def compare_noise():
    model=Wrapper()
    #acc=test_zero_shot_classfication(model=model)
    dataset=datasets.load_from_disk(Params["train_dataset_path"],keep_in_memory=True)['test']
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
    dataset=datasets.load_from_disk(Params["train_dataset_path"],keep_in_memory=True)['test']
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
    model=CLIPModel.from_pretrained(Params["model_name"]).to(Params['device'])
    dataset=datasets.load_from_disk(Params["train_dataset_path"],keep_in_memory=True)['test']
    dataset.set_format('torch',columns=['image','caption'])
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
    def collate_fn(x: List[dict]) -> BatchEncoding:
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
def test_loss():
    model=Wrapper()
    dataset=datasets.load_from_disk(Params["train_dataset_path"],keep_in_memory=True)['test']
    dataset.set_format('torch',columns=['image','caption'])
    loader=CLIPLoader(dataset, batch_size=Params['batch_size'], num_workers=Params['num_workers'], shuffle=True,
                              collate_fn=None, pin_memory_device="cuda:0", pin_memory=True)
    model.eval()
    for para in model.model.gen.parameters():
        para.data=para.data.float()
    for para in model.model.target.parameters():
        para.data=para.data.float()
    data=next(iter(loader)).to(Params['device']).float()
    y=model(data)
    loss=model.loss(y)
    print(loss)
if __name__== "__main__":
    train_vae_model(True)
    """
    dataset=datasets.load_from_disk(Params["train_dataset_path"],keep_in_memory=True)['test']
    print(dataset)
    dataset.set_format("torch")
    image=dataset[0]['image'].float().unsqueeze(0).cuda()/255.
    model=Wrapper(image=image)
    model.model.train()
    outputs=model.forward()
    loss,_=model.loss(outputs)
    print(loss)
    loss.backward()
    model.optim.step()
    model.optim.zero_grad()
    outputs=model.forward()
    loss,_=model.loss(outputs)
    print(loss)
    loss.backward()
    print(model.model.input_img.grad)
    """
