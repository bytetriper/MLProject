import torch

if __name__=="__main__":
    tens=torch.tensor([1.1])
    print(tens.dtype)
    halftens=tens.half()
    print(halftens.dtype)
    print((tens+halftens).dtype)