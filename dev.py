from diffusers import AutoencoderKL
import datasets
from transformers import CLIPVisionModelWithProjection,BatchEncoding,BatchFeature
import torch
from Params import Params
def etst():
    model = UNet2DModel(
    sample_size=224,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
    model.train()
    model.cuda()
    torch.cuda.empty_cache()
    x=torch.randn(1,3,224,224).cuda()
    y=model(x,0)
    print(y['sample'].shape)
def encode(x:torch.Tensor,vae:AutoencoderKL)->torch.Tensor:
    x=x.float()/255.
    x=x*2-1
    return 0.18215*vae.encode(x).latent_dist.sample()
def decode(latent:torch.Tensor,vae:AutoencoderKL)->torch.Tensor:
    latent=latent/0.18215
    dec=vae.decode(latent)['sample']
    dec = (dec/2 + 0.5).clamp(0, 1)
    return dec
def plt_img(img:torch.Tensor,save_path:str):
    #use PIL to plot image
    import PIL
    from torchvision import transforms
    if img.dtype==torch.int64:
        img=img.float()/255.
    img=img.cpu()
    pil_img=transforms.transforms.ToPILImage()(img)
    pil_img.save(save_path)
    
if __name__=="__main__":
    
    vae=AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-base",subfolder="vae")
    vae.eval()
    vae.cuda()
    dataset=datasets.load_from_disk(Params["train_dataset_path"],keep_in_memory=True)['test']
    print(dataset)
    dataset.set_format("torch")
    img=dataset[128]['image']
    plt_img(img,"/root/autodl-tmp/fool_clip/imgs/source.png")
    img=img.unsqueeze(0).cuda()
    enc=encode(img,vae)
    enc=enc+torch.randn_like(enc)*0.01
    dec=decode(enc,vae)
    dec=dec.detach()
    img=img.float()/255
    print("l1:",(torch.abs(img-dec).mean()))
    l2diss=torch.nn.MSELoss()
    print("l2:",l2diss(img,dec))
    print('linf:',torch.max(torch.abs(img-dec)))
    plt_img(dec[0],"/root/autodl-tmp/fool_clip/imgs/noised.png")
    

