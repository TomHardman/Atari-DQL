import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

class LossBuffer:
    def __init__(self, plot_freq=5000) -> None:
        self.memory = []
        self.avg = 0
        self.plot_freq = plot_freq

    def push(self, loss):
        if loss == None:
            return
        
        self.memory.append(loss)
        self.avg = self.avg * (len(self) - 1)/len(self) + loss/len(self)

        if len(self.memory) % self.plot_freq == 0:
            self.plot_loss(len(self.memory))
    
    def plot_loss(self, iters):
        plt.cla()
        plt.plot(self.memory)
        plt.yscale('log')
        plt.savefig(f'loss_{iters}.png')
    
    def __len__(self):
        return len(self.memory)


def process_image(image: list, device, crop_top=None, crop_bottom=None, crop_left=None, crop_right=None,
                  downsample_factor: int=2, grayscale=False) -> torch.Tensor: 
    """
    Takes image in shape rows x columns x channels, crops, then downsamples by a factor of 2
    """

    with record_function("image_convert"):
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
    
    with record_function("aa_crop_permute"):
        if len(image.shape) == 3:
            image = image[crop_top:crop_bottom, crop_left:crop_right, :]
            image = image.permute(2, 0, 1)
        else:
            image = image[crop_top:crop_bottom, crop_left:crop_right]
            image = image.unsqueeze(0)

    with record_function("aa_interpolate"):
        image = F.interpolate(image.unsqueeze(0), 
                          scale_factor=1/downsample_factor, 
                          mode='bilinear', 
                          align_corners=False).squeeze(0)
    
    return image