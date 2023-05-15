from net import Fool_CLip,Wrapper
from test import test_zero_shot_classfication,get_visualization
from evaluation import test_on_classifier
from sys import argv
from Params import Params
from train import train_model
import torch
import datasets
if __name__ == '__main__':
   # main()
   #model=Wrapper()
   #test_zero_shot_classfication(model)
   len=len(argv)
   if len==1:
      print("Usage: python main.py [train|test|vis(visualization))|eval]")
      exit(0)
   print('using device:',Params['device'])
   print('using pre-trained target model:',Params['model_name'])
   model=Wrapper()
   if argv[1]=="train":
      dataset=datasets.load_from_disk(Params['train_dataset_path'])
      train_model(model,dataset)
   elif argv[1]=="test":
      if len==2:
         print("Usage: python main.py test [zero_shot|trained]")
         exit(0)
      if argv[2]=="zero_shot":
         test_zero_shot_classfication()
      elif argv[2]=="trained":
         test_zero_shot_classfication(model)
   elif argv[1]=="vis":
      get_visualization(model=model)
   elif argv[1]=="eval":
      if len==2:
         print("Usage: python main.py eval [dft(default)|trained]")
         exit(0)
      if argv[2]=="dft":
         test_on_classifier()
      elif argv[2]=="trained":
         if not Params['load_generator']:
            print("Params['load_generator'] is set to False, automatically loading from default path:"+Params['generator_path'])
            model.gen.load_state_dict(torch.load(Params['generator_path']))
         test_on_classifier(model.gen)