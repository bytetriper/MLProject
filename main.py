from net import Fool_CLip,Wrapper
from test import test_zero_shot_classfication,get_visualization
from evaluation import test_on_classifier
from sys import argv
from Params import Params
from train import train_model,train_vae_model
import torch
import datasets
import os
from apex import amp
import argparse
# add a with_gen,logdir arg

parser=argparse.ArgumentParser()
parser.add_argument('-train',dafault=False,type=bool)
parser.add_argument('-eval',default=False,type=bool)
parser.add_argument('-vis',default=False,type=bool)

parser.add_argument('-with_gen',default=False,type=bool)
parser.add_argument('-logdir',default=None,type=str)
if __name__ == '__main__':
   # main()
   #model=Wrapper()
   #test_zero_shot_classfication(model)
   args=parser.parse_args()
   Params['with_gen']=args.with_gen
   behaviour='train' if args.train else 'vis' if args.vis else 'eval' if args.eval else None
   print('using device:',Params['device'])
   print('using pre-trained target model:',Params['model_name'])
   model=Wrapper()
   if behaviour =="train":
      if Params['with_gen']:
         logdir=args.logdir
         if logdir is None:
            print("Usage: python main.py train [logdir]")
            exit(0)
         runs_folder=os.path.join(Params['summary_path'],'runs/'+argv[2])
         dataset=datasets.load_from_disk(Params['train_dataset_path'])
         train_model(model,dataset,runs_folder)
      else:
         train_vae_model(True)
   elif behaviour=="vis":
      get_visualization(model=model.model)
   elif behaviour=="eval":
      if len==2:
         print("Usage: python main.py eval [dft(default)|trained]")
         exit(0)
      if argv[2]=="dft":
         test_on_classifier()
      elif argv[2]=="trained":
         path=Params['base_path'] if Params['base']=='base' else Params['unet_path']
         print("Loading model from",path)
         if Params['amp_mode']:
            model.load()
            for para in model.model.gen.parameters():
               para.data=para.data.float()
         else:
            model.model.gen.load_state_dict(torch.load(path)['gen'])
         test_on_classifier(model.model)