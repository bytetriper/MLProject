from net import Wrapper
from Params import Params
import datasets
from utils import *
from matplotlib import pyplot as plt
from apex import amp
from random import randint
from dev import plt_img
from torchvision import transforms
import torch.functional as F

def train_model(model: Wrapper, dataset: datasets.DatasetDict, summary_folder: str) -> None:
    train_set = dataset['train']
    # remove 'image' 'caption' from the dataset
    train_set.set_format('torch', columns=['image', 'caption'])
    train_loader = CLIPLoader(train_set, batch_size=Params['batch_size'], num_workers=Params['num_workers'], shuffle=True,
                              collate_fn=None, pin_memory_device="cuda:0", pin_memory=True)
    loss_history = model.train(train_loader, summary_folder)
    plt.plot(loss_history)
    plt.savefig('loss.png')


def train_vae_model(with_label: bool = False) -> None:
    if with_label:
        dataset = load_data_with_retry(
            Params['transfer_dataset_path'], keep_in_memory=True)['train']

    else:
        dataset = load_data_with_retry(
            Params['train_dataset_path'], keep_in_memory=True)['test']
    dataset.set_format('torch')
    idx = randint(0, len(dataset)-1)
    img = dataset[idx]['image'].float()/255.
    if with_label:
        img = img.permute(2, 0, 1)
    print('plotting image in ./imgs/source.png')
    plt_img(img, './imgs/source.png')
    img = img.cuda().unsqueeze(0)
    model = Wrapper(image=img)
    if with_label:
        pred_label=torch.softmax(model.model.original_outputs['last_hidden_state'],dim=1).argmax()
        gt_label=dataset[idx]['label']
        print('pred_label:',pred_label)
        print('gt_label:',gt_label)
    noised = model.gradient_ascent(
        100, label=None if with_label == False else dataset[idx]['label'].unsqueeze(0).cuda())[0]
    print('plotting noised image in ./imgs/noised.png')
    plt_img(noised, './imgs/noised.png')
    # calculate l1,l2,linf distance
    img = img.squeeze(0)
    print('l1:', (torch.abs(img-noised).mean()))
    l2diss = torch.nn.MSELoss()
    print('l2:', l2diss(img, noised))
    print('linf:', torch.max(torch.abs(img-noised)))
    # save the noise
    noise = noised-img
    print('plotting noise in ./imgs/noise.png')
    plt_img(noise, './imgs/noise.png')


if __name__ == '__main__':
    model = Wrapper()
    dataset = load_data_with_retry(
        Params['train_dataset_path'], keep_in_memory=True)
    train_model(model, dataset)
