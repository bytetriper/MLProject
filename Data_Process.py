
import datasets
from torchvision import transforms
import torch
def test_map():
    Dats=datasets.load_dataset('nlphuji/flickr30k')['test']
    #remove the column 'sentids', 'split', 'img_id', 'filename' of the dataset
    Dats=Dats.remove_columns(['sentids','img_id','split','filename'])
    #map the Dats into 'img','txt'
    Transform=transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.ToTensor()
    ])
    Dats=Dats.train_test_split(test_size=0.001)['test']
    Dats=Dats.map(lambda x: {'image':Transform(x['image']),'txt':x['caption']})
    Dats.set_format('torch')
    print(Dats)
    print(Dats[0]['image'].shape)
def make_dataset():
    Dats=datasets.load_dataset('nlphuji/flickr30k',keep_in_memory=True)['test']
    #remove the column 'sentids', 'split', 'img_id', 'filename' of the dataset
    Dats=Dats.remove_columns(['sentids','img_id','split','filename'])
    #map the Dats into 'img','txt'
    Transform=transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.ToTensor()
    ])
    print(Dats)
    Dats=Dats.map(lambda x: {'image':Transform(x['image']),'caption':x['caption']},num_proc=5)
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
if __name__ == "__main__":
    make_dataset()
    #test_map()