from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as tvF
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import albumentations as A
from sys import exit
import math
from tps.numpy import warp_image_cv

def rot_crop(x):
    """return maximum width ratio of rotated image without letterbox"""
    x = abs(x)
    deg45 = math.pi * 0.25
    deg135 = math.pi * 0.75
    x = x * math.pi / 180
    a = (math.sin(deg135 - x) - math.sin(deg45 - x))/(math.cos(deg135-x)-math.cos(deg45-x))
    return math.sqrt(2) * (math.sin(deg45-x) - a*math.cos(deg45-x)) / (1-a)

class RandomFRC(T.RandomResizedCrop):
    """RandomHorizontalFlip + RandomRotation + RandomResizedCrop 2 images"""
    def __call__(self, img1, img2):
        img1 = tvF.resize(img1, self.size, interpolation=Image.LANCZOS)
        img2 = tvF.resize(img2, self.size, interpolation=Image.LANCZOS)
        if random.random() < 0.5:
            img1 = tvF.hflip(img1)
            img2 = tvF.hflip(img2)
        if random.random() < 0.5:
            rot = random.uniform(-10, 10)
            crop_ratio = rot_crop(rot)
            img1 = tvF.rotate(img1, rot, resample=Image.BILINEAR)
            img2 = tvF.rotate(img2, rot, resample=Image.BILINEAR)
            img1 = tvF.center_crop(img1, int(img1.size[0] * crop_ratio))
            img2 = tvF.center_crop(img2, int(img2.size[0] * crop_ratio))

        i, j, h, w = self.get_params(img1, self.scale, self.ratio)

        # return the image with the same transformation
        return (tvF.resized_crop(img1, i, j, h, w, self.size, self.interpolation),
                tvF.resized_crop(img2, i, j, h, w, self.size, self.interpolation))

class Dataset(data.Dataset):
    def __init__(self, image_dir, line_dir, transform_common, transform_a, transform_line, transform_original):
        
        self.image_dir = image_dir
        self.line_dir = line_dir
        
        self.transform_common = transform_common
        self.transform_a = transform_a
        self.transform_line = transform_line
        self.transform_original = transform_original
        
        self.ids = [f.split('/')[-1] for f in glob(os.path.join(line_dir, '*.png'))]
    
    def __getitem__(self, index):
        filename = self.ids[index]
      
        image_path = os.path.join(self.image_dir, filename)
        line_path = os.path.join(self.line_dir, filename)

        image = Image.open(image_path).convert('RGB')
        line = Image.open(line_path).convert('L')
    
        image_ori, line = self.transform_common(image, line)
        
        I_original = self.transform_original(image_ori)
        
        I_gt = self.transform_a(image_ori)
        
        line = self.transform_line(line)
        
        if random.random() <= 0.9:
            # I_r = TPS(I_gt)
            # I_r = TPS(I_gt.unsqueeze(0)).squeeze()
            I_r = I_gt.clone()
        else:
            I_r = torch.zeros(I_gt.size())

        return I_original, I_gt, I_r, line
        
        
    def __len__(self):
        return len(self.ids)

# def TPS(x):
#     c,h,w = x.size()
#     x = x.numpy()
#     common = np.random.rand(4, 2)
#     A = np.random.rand(2, 2)
#     B = np.random.rand(2, 2)
#     c_src = np.concatenate([common, A], 0)
#     c_dst = np.concatenate([common, B], 0)
#     warped = warp_image_cv(x.transpose(2, 1, 0), c_src, c_dst, dshape=(h, w)).transpose((2, 0, 1)) # HWC -> CHW
#     return torch.from_numpy(warped)


def TPS(x):
    
    def affine_transform(x, theta):
        theta = theta.view(-1, 2, 3)
        # grid = F.affine_grid(theta, x.size(), align_corners=True)
        # x = F.grid_sample(x, grid, align_corners=True)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
    
    theta1 = np.zeros(9)
    theta1[0:6] = np.random.randn(6) * 0.15
    theta1 = theta1 + np.array([1,0,0,0,1,0,0,0,1])
    affine1 = np.reshape(theta1, (3,3))
    affine1 = np.reshape(affine1, -1)[0:6]
    affine1 = torch.from_numpy(affine1).type(torch.FloatTensor)
    x = affine_transform(x, affine1) # source image
    
    return x


def get_loader(crop_size=256, image_size=266, batch_size=16, dataset='CelebA', mode='train', num_workers=8, line_type='xdog', ROOT='./datasets'):
    """Build and return a data loader."""
    transform_common = []
    transform_a = []
    transform_line = []
    transform_original = []
    
    transform_common = RandomFRC(crop_size, scale=(0.9, 1.0), ratio=(0.95, 1.05), interpolation=Image.LANCZOS)

    transform_a = T.Compose([
        T.ColorJitter(brightness=0.05, contrast=0.2, saturation=0.4, hue=0.4),
        T.Resize((crop_size, crop_size), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
        
    transform_line = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.Resize((crop_size, crop_size), interpolation=Image.LANCZOS),
        T.ToTensor(),
        # T.RandomErasing(p=0.9, value=1., scale=(0.02, 0.1))
    ])
    
    transform_original = T.Compose([
        T.Resize((crop_size, crop_size), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    
    if dataset == 'line_art':
        image_dir = os.path.join(ROOT, 'line_art/train/color')
        line_dir = os.path.join(ROOT, 'line_art/train/xdog')
    elif dataset == 'tag2pix':
        image_dir = os.path.join(ROOT, 'tag2pix/rgb_cropped')
        if line_type == 'xdog':
            line_dir = os.path.join(ROOT, 'tag2pix/xdog_train')
        elif line_type == 'keras':
            line_dir = os.path.join(ROOT, 'tag2pix/keras_train')
        
    dataset = Dataset(image_dir, line_dir,
                        transform_common, transform_a, transform_line, transform_original)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader

if __name__ == '__main__':
    
    def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    from torchvision.utils import save_image
    loader = get_loader(
        crop_size=256,
        image_size=256,
        batch_size=20,
        dataset='tag2pix', mode='test', num_workers=4, line_type='xdog', 
        ROOT='./data'
    )
    
    loader = iter(loader)
    
    I_ori, I_gt, I_r, I_s = next(loader)
    
    I_concat = denorm(torch.cat([I_ori, I_gt, I_r], dim=2))
    I_concat = torch.cat([I_concat, I_s.repeat(1,3,1,1)], dim=2)
    
    save_image(I_concat, 'tmp.png')