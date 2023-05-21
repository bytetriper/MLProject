from typing import Any
from net import Wrapper, Fool_CLip
from Params import Params
from torch import nn as nn
import torch
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from transformers import ResNetForImageClassification,ConvNextForImageClassification,ViTForImageClassification,FocalNetForImageClassification
import timm
from torchmetrics import StructuralSimilarityIndexMeasure as ssim
test_save_path = "./models/test_resnet.pth"


class Test_Resnet(nn.Module):
    def __init__(self) -> None:
        super(Test_Resnet, self).__init__()
        self.net = ResNetForImageClassification.from_pretrained("microsoft/resnet-101")
        self.net.eval()
        self.psc=Image_Net_Constants()
    def forward(self, x ,output_hidden_states:bool=False):
        x=self.psc(x)
        data=self.net(x, output_hidden_states=output_hidden_states)
        if output_hidden_states:
            return data['hidden_states'],data['logits']
        return data['logits']

class Test_Vgg(nn.Module):
    def __init__(self) -> None:
        super(Test_Vgg, self).__init__()
        #self.net =AutoModelForImageClassification.from_pretrained("google/inception_v3")
        self.net=timm.create_model('vgg19', pretrained=True)
        self.net.eval()
        self.psc=Image_Net_Constants()

    def forward(self, x,output_hidden_states=False):
        x=self.psc(x)
        data=self.net(x, output_hidden_states=output_hidden_states)
        if output_hidden_states:
            return data['hidden_states'],data['logits']
        return data['logits']
class Test_ConvNext(nn.Module):
    def __init__(self) -> None:
        super(Test_ConvNext, self).__init__()
        self.net = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224")
        self.net.eval()
        self.psc=Image_Net_Constants()

    def forward(self, x ,output_hidden_states:bool=False):
        x=self.psc(x)
        data=self.net(x, output_hidden_states=output_hidden_states)
        if output_hidden_states:
            return data['hidden_states'],data['logits']
        return data['logits']
class Test_FocalNet(nn.Module):
    def __init__(self) -> None:
        super(Test_FocalNet, self).__init__()
        self.net = FocalNetForImageClassification.from_pretrained("microsoft/focalnet-base")
        self.net.eval()
        self.psc=Image_Net_Constants()
    def forward(self, x,output_hidden_states=False):
        x=self.psc(x)
        data=self.net(x, output_hidden_states=output_hidden_states)
        if output_hidden_states:
            return data['hidden_states'],data['logits']
        return data['logits']
class Test_ViT(nn.Module):
    def __init__(self) -> None:
        super(Test_ViT, self).__init__()
        self.net=ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.net.eval()
        self.psc=Image_Net_Constants()

    def forward(self, x,output_hidden_states=False):
        x=self.psc(x)
        data=self.net(x, output_hidden_states=output_hidden_states)
        if output_hidden_states:
            return data['hidden_states'],data['logits']
        return data['logits']
class QuickTrain():
    def __init__(self, model: nn.Module, target_model=None, load: bool = True) -> None:
        self.model = model
        if load:
            self.model.net.load_state_dict(torch.load(test_save_path))
        self.target_model = target_model
        if self.target_model != None:
            self.target_model.cuda()
        self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=5e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.9)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, epoch: int, saved: bool):
        for i in range(epoch):
            self.model.train()
            tbar = tqdm(train_loader)
            tbar.set_description(f"Epoch {i}")
            avg_loss = 0
            total_loss = 0
            for batch in train_loader:
                img = batch['image'].float().permute(0, 3, 1, 2).cuda()/255.
                label = batch['label'].cuda()
                img = img.cuda()
                label = label.cuda()
                logits = self.model(img)
                loss = self.criterion(logits, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tbar.update(1)
                total_loss += loss.item()
                avg_loss = total_loss/(tbar.n+1)
                tbar.set_postfix(
                    loss=avg_loss, lr=self.optimizer.param_groups[0]['lr'])
            self.scheduler.step()
            self.eval(valid_loader)
            if saved:
                torch.save(self.model.net.state_dict(), test_save_path)

    def eval(self, valid_loader: DataLoader):
        self.model.eval()
        if self.target_model != None:
            self.target_model.eval()
        tbar = tqdm(valid_loader)
        tbar.set_description(f"Eval")
        correct = 0
        total = 0
        preprocessor=Image_Net_Constants()
        with torch.no_grad():
            for batch in valid_loader:
                img = batch['image'].cuda()
                label = batch['label'].cuda()
                img = img.cuda()
                label = label.cuda()
                if self.target_model != None:
                    img,noise= self.target_model.gen(img)
                #img=preprocessor(img)
                logits= self.model(img)
                prob = torch.softmax(logits, dim=1)
                pred = torch.argmax(prob, dim=1)
                correct += (pred == label).sum().item()
                total += len(pred)
                tbar.update(1)
            print(f"\nAccuracy: {correct/total}\n")


def quick_train(batch_size: int = 256, epoch: int = 10):
    dataset = datasets.load_from_disk(
        Params["transfer_dataset_path"], keep_in_memory=True)
    dataset = dataset['train'].train_test_split(test_size=0.2)
    train_set = dataset['train']
    valid_set = dataset['test']
    train_set.set_format(type='torch', columns=['image', 'label'])
    valid_set.set_format(type='torch', columns=['image', 'label'])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=15, pin_memory=True, pin_memory_device="cuda:0")
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=15)
    trainer = QuickTrain(Test_Resnet(), load=True)
    trainer.train(train_loader, valid_loader, epoch, True)
    trainer.eval(valid_loader)

class TEST_PIPELINE():
    def __init__(self) -> None:
        self.disc=Test_ViT()
        self.model=Wrapper(
            gen=None,
            target=self.disc
        )
        self.oriimg_save_path=Params["original_image_path"]
        self.ascimg_save_path=Params['ascented_image_path']
        self.difimg_save_path=Params['difference_image_path']
    def CEloss(self,**kwargs):# aloss for huggingface resnet-50
        outputs=kwargs["outputs"]
        labels=kwargs["labels"]
        return nn.CrossEntropyLoss()(outputs,labels)
    def SSIMLoss(self,**kwargs):
        if not hasattr(self,"ssim"):
            self.ssim=ssim(data_range=1.).to(Params["device"])
        noised=kwargs["noised"]
        image=kwargs["image"]
        return self.ssim(noised,image)
    def CombinedLoss(self,**kwargs):
        return self.CEloss(**kwargs)+50*self.SSIMLoss(**kwargs)
    def DALoss(self,**kwargs):
        noised_feature=kwargs["noised_feature"]
        ori_feature=kwargs["image_feature"]
        #both list
        if not hasattr(self,"coslossf"):
            self.coslossf=nn.CosineSimilarity(dim=2)
        loss=0

        [loss:=loss+self.coslossf(nf,of).mean() for nf,of in zip(noised_feature,ori_feature)]
        loss/=len(noised_feature)
        return 1-loss
    def AutoLoss(self,**kwargs):
        outputs=kwargs["outputs"]
        image=kwargs["image"]
        noised=kwargs["noised"]
        label=kwargs["labels"]
        if isinstance(outputs,tuple):
            #use DA loss
            return self.DALoss(noised_feature=outputs[0],image_feature=image[0])
        else:
            #use CE loss
            return self.CEloss(outputs=outputs,labels=label)
    def RUN_CE_TEST(self,epoch:int, image:torch.Tensor,label:torch.Tensor)->torch.Tensor:
        print("RUNNING CE TEST")
        self.model.init_features(image.to(Params["device"]))
        noised,loss=self.model.gradient_ascent(
            epoch=epoch,
            loss_fn=self.AutoLoss,
            attached_fn=self.SSIMLoss,
            label=label,
            #PGD_eps=0.1,
            output_hidden=False,
        )
        return noised,loss
    def RUN_PGD_TEST(self,epoch:int, image:torch.Tensor,label:torch.Tensor)->torch.Tensor:
        print("RUNNING PGD TEST")
        self.model.init_imgs(image.to(Params["device"]))
        noised,loss=self.model.PGD_direct_attack(
            epoch=epoch,
            loss_fn=self.AutoLoss,
            attached_fn=self.SSIMLoss,
            label=label,
            eps=16/255,
            output_hidden=False,
        )
        return noised,loss
    def MAKE_NOISED_DATASET(self,epoch:int,dataset:datasets.Dataset)->datasets.Dataset:
        transform=transforms.transforms.Compose([
                transforms.transforms.Resize((224,224)),
                transforms.transforms.PILToTensor(),
                transforms.transforms.ConvertImageDtype(torch.float32),
            ])
        def map_fn(batch):
            image=batch["image"]
            label=batch["label"]
            #turn image form list to tensor
            image=[transform(img.convert('RGB')) for img in image]
            label=torch.tensor(label)
            image=torch.stack(image,dim=0)
            noised,_=self.RUN_PGD_TEST(epoch,image.cuda().clone(),label.cuda().clone())
            return {
                "image":image,
                "label":label,
                "noised":noised.cpu()
            }
        return dataset.map(map_fn,batched=True,batch_size=60,num_proc=1)
    def measure_dis(self,noise:torch.Tensor, image:torch.Tensor)->torch.Tensor:
        if not hasattr(self,"l1_fn"):
            self.l1_fn=nn.L1Loss()
        if not hasattr(self,"l2_fn"):
            self.l2_fn=nn.MSELoss()
        if not hasattr(self,"linf_fn"):
            self.linf_fn=torch.max
        l1=self.l1_fn(noise,image)
        l2=self.l2_fn(noise,image)
        linf=self.linf_fn((noise-image).abs())
        return l1,l2,linf
    def RUN_TEST_ON_ALL(self,epoch:int)->None:
        loader=ImageNet_Loader(
            batch_size=60,
            shuffle=True,
            num_workers=15,
            pin_memory=True,
            pin_memory_device="cuda:0"
        )
        total=0
        loss_list=[(0,0)]*epoch
        for data in loader:
            torch.cuda.empty_cache()
            image=data["image"].to(Params["device"])
            label=data["label"].to(Params["device"])
            _,loss=self.RUN_PGD_TEST(epoch,image.clone(),label.clone())
            loss_list=[(l[0]+ll[0],l[1]+ll[1]) for l,ll in zip(loss_list,loss)]
            total+=1
        loss_list=[(l[0]/total,l[1]/total) for l in loss_list]
        
        #use pickle to save the loss_list
        import pickle
        with open("PGD_ViT.pkl","wb") as f:
            pickle.dump(loss_list,f)
        for i,l in enumerate(loss_list):
            print(i,l[0].item(),l[1])
    def __call__(self,  **kwds: Any ) -> None:
        # assert that image and label are in kwds
        print('running test pipeline, get args:',kwds)
        if "loader" in kwds.keys():
            loader= kwds["loader"]
        else:
            loader=ImageNet_Loader(
                batch_size=12,
                shuffle=False,
                num_workers=15,
                pin_memory=True,
                pin_memory_device="cuda:0"
            )
        if "image" in kwds.keys() and "label" in kwds.keys():
            image=kwds["image"]
            label=kwds["label"]
        else:
            data=(next(iter(loader)))
            image,label=data['image'].to(Params['device']),data['label'].to(Params['device'])
        print('img:',image.shape)
        print('label:',label.shape)
        print('loader:',loader)
        print("Running test pipeline")        
        pred=self.disc(image)
        pred_label=pred.argmax(dim=1)
        acc=(pred_label==label).float().mean()
        noised,_=self.RUN_CE_TEST(epoch=30,image=image,label=label)
        #noised,_=self.RUN_PGD_TEST(epoch=30,image=image.clone(),label=label.clone())
        predn=self.disc(noised)
        pred_labeln=predn.argmax(dim=1)
        print('orginal acc:',acc)
        print("label:",label)
        fool_acc=(pred_labeln==label).float().mean()
        print('fooling rate:',(acc-fool_acc)/acc)
        l1,l2,lf=self.measure_dis(noised,image)
        simloss=self.SSIMLoss(noised=noised,image=image)
        print("l1:",l1)
        print("l2:",l2)
        print("linf:",lf)
        print("ssim:",simloss)
        if "plt" in kwds:
            plt=kwds["plt"]
            if plt==True:
                print("Plotting ascented image")
                plt_imgs(noised,self.ascimg_save_path)
                plt_imgs(image,self.oriimg_save_path)
        if "plt_dif" in kwds:
            plt_dif=kwds["plt_dif"]
            if plt_dif==True:
                print("Plotting diffed image")
                plt_difs(noised,image,self.difimg_save_path)
        print("Test pipeline finished")
        return noised
        
def test_on_classifier(target_model: nn.Module = None, batch_size: int = 512):
    print("Testing on classifier on transfer dataset:tiny-imagenet with batch_size=", batch_size)
    #dataset = datasets.load_from_disk(
    #    Params["transfer_dataset_path"], keep_in_memory=True)['train']
    loader=ImageNet_Loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=15,
        pin_memory=True,
        pin_memory_device="cuda:0"
    )
    model = Test_Vgg().cuda()
    if target_model == None:
        print("Using default model(None) as generator")
    else:
        print("Using custom model as generator")
    trainer = QuickTrain(model, target_model, load=False)
    trainer.eval(loader)

def test_on_noised_dataset(model:nn.Module=None):
    loader=ImageNet_Loader(noised=True,
                           batch_size=26,
                           shuffle=True,
                            num_workers=15,
                            pin_memory=True,
                            pin_memory_device="cuda:0")
    disc=Test_ViT().cuda() if model is None else model.cuda()
    corrected=0
    vanilla_corrected=0
    total=0
    transform=Image_Net_Constants()
    avg_loss=0
    total_loss=0
    SSIMLOSS=ssim(data_range=1.).to(Params['device'])
    for i,data in enumerate(loader):
        image,noised,label=data['image'].cuda(),data['noised'].cuda(),data['label'].cuda()
        pred_labelv=disc(image).argmax(dim=1)
        pred=disc(noised)
        pred_label=pred.argmax(dim=1)
        corrected+=(pred_label==label).float().sum()
        vanilla_corrected+=(pred_labelv==label).float().sum()
        total+=label.shape[0]
        loss=SSIMLOSS(noised,image)
        total_loss+=loss.item()
    acc=corrected/total
    avg_loss=total_loss/len(loader)
    print("acc:",acc)
    vacc=vanilla_corrected/total
    print("vanilla acc:",vacc)
    print("delta acc:",(vacc-acc)/vacc)
    print("\n")
    #fool_rate=(acc-vacc)/acc
    #print("avg loss:",avg_loss)



if __name__ == "__main__":
    # test_on_classifier()
    #quick_train(epoch=5)
    #test_on_classifier(batch_size=64)
    #models=[Test_ViT(),Test_Resnet(),Test_FocalNet(),Test_ConvNext()]
    #for model in models:
    #    test_on_noised_dataset(model)
    PIPl=TEST_PIPELINE()
    #PIPl.RUN_TEST_ON_ALL(30)
    #PIPl.MAKE_NOISED_DATASET(
    #    epoch=30,
    #)
    #PIPl.RUN_TEST_ON_ALL(30)
    PIPl(plt=True)
    #model=Test_Inc()