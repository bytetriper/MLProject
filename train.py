from net import Wrapper
from Params import Params
import datasets
from utils import *
from matplotlib import pyplot as plt
def train_model(model: Wrapper, dataset: datasets.DatasetDict)->None:
    train_set = dataset['train']
    # remove 'image' 'caption' from the dataset
    train_set.set_format('torch', columns=['image', 'caption'])
    train_loader = CLIPLoader(train_set, batch_size=Params['batch_size'], num_workers=Params['num_workers'], shuffle=True,
                              collate_fn=None, pin_memory_device="cuda:0", pin_memory=True)
    loss_history = model.train(train_loader)
    plt.plot(loss_history)
    plt.savefig('loss.png')

if __name__=='__main__':
    model=Wrapper()
    dataset=load_data_with_retry(Params['train_dataset_path'],keep_in_memory=True)
    train_model(model,dataset)