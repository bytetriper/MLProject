from net import Wrapper, Fool_CLip
from Params import Params
from torch import nn as nn
import torch
import datasets
from torchvision.models.resnet import ResNet50_Weights
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
test_save_path = "./models/test_resnet.pth"


class Test_Resnet(nn.Module):
    def __init__(self) -> None:
        super(Test_Resnet, self).__init__()
        self.net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    def forward(self, x):
        return self.net(x)


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
        with torch.no_grad():
            for batch in valid_loader:
                img = batch['image'].float().permute(0, 3, 1, 2).cuda()/255.
                label = batch['label'].cuda()
                img = img.cuda()
                label = label.cuda()
                tmp = img
                if self.target_model != None:
                    img,noise= self.target_model.gen(img)
                logits,_ = self.model(img)
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


def test_on_classifier(target_model: nn.Module = None, batch_size: int = 512):
    print("Testing on classifier on transfer dataset:tiny-imagenet with batch_size=", batch_size)
    dataset = datasets.load_from_disk(
        Params["transfer_dataset_path"], keep_in_memory=True)['train']
    dataset.set_format(type='torch', columns=['image', 'label'])
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=15)
    model = Test_Resnet()
    if target_model == None:
        print("Using default model(None) as generator")
    else:
        print("Using custom model as generator")
    trainer = QuickTrain(model, target_model, load=True)
    trainer.eval(loader)


if __name__ == "__main__":
    # test_on_classifier()
    quick_train(epoch=5)
