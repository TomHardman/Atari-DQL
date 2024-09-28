import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch

class Buffer:
    def __init__(self, ) -> None:
        self.short_memory = []
        self.long_memory = []
        self.avg = 0

    def push(self, loss):
        if loss == None:
            return
        
        self.short_memory.append(loss)
        self.avg = self.avg * (len(self) - 1)/len(self) + loss/len(self)
    
    def update_long(self):
        self.long_memory.append(self.avg)
        self.avg = 0
        self.short_memory = []
    
    def plot_loss(self, path, log_scale=False):
        plt.cla()
        plt.plot(self.long_memory)
        if log_scale:
            plt.yscale('log')
        plt.xlabel('Epochs')
        plt.savefig(path)
    
    def __len__(self):
        return len(self.short_memory)
    

def to_tensor(image_stack: np.ndarray, device) -> torch.Tensor:
    """
    Converts numpy image stack to torch tensor
    """
    image_stack = torch.from_numpy(np.asarray(image_stack).astype(np.float32)).to(device)
    return image_stack

def process_image(image: list, device, crop_top=None, crop_bottom=None, crop_left=None, crop_right=None,
                  downsample_factor: int=2, grayscale=False) -> torch.Tensor: 
    """
    Takes image in shape rows x columns x channels, crops, then downsamples by a factor of 2
    """

    image = image.astype(np.float32)
    image = torch.from_numpy(image).to(device)

    if crop_top is None:
        crop_top = 0
    if crop_bottom is None:
        crop_bottom = image.shape[0]
    if crop_left is None:
        crop_left = 0
    if crop_right is None:
        crop_right = image.shape[1]
    
    if len(image.shape) == 3:
        image = image[crop_top:crop_bottom, crop_left:crop_right, :]
        image = image.permute(2, 0, 1)
    else:
        image = image[crop_top:crop_bottom, crop_left:crop_right]
        image = image.unsqueeze(0)
    
    image = F.interpolate(image.unsqueeze(0), 
                        scale_factor=1/downsample_factor, 
                        mode='bilinear', 
                        align_corners=False).squeeze(0)
    return image
