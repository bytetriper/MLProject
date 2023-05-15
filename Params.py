import torch
Params = {
    'lr': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_generator': True,
    'load_generator': False,
    'generator_path': '/root/autodl-tmp/fool_clip/models/generator.pth',
    'model_name': 'openai/clip-vit-base-patch16',
    'batch_size': 32,
    'epochs': 5,
    'select_col': 1,  # should be in range [0-4]
    'num_workers': 5,
    "noise_bound": 16/255,
    "train_dataset_path": "/root/autodl-tmp/fool_clip/train_dataset",
    "test_dataset_path": "/root/autodl-tmp/fool_clip/test_dataset",
    "transfer_dataset_path": "/root/autodl-tmp/fool_clip/tiny-imagenet",
    "summary_path": "/root/autodl-tmp/fool_clip/summary",
    "original_image_path": "/root/autodl-tmp/fool_clip/imgs/source.png",
    "noised_image_path": "/root/autodl-tmp/fool_clip/imgs/noised.png",
    "noise_image_path": "/root/autodl-tmp/fool_clip/imgs/noise.png",
    "noise_path": "/root/autodl-tmp/fool_clip/data/noise.npy",

}