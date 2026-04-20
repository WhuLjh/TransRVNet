import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation_adapt
import cv2

class Mydata_adapt(Dataset):
    """
    num_classes: 2
    """
    CLASSES = ['background', 'rooftop_vegetation']
    PALETTE = torch.tensor([[0, 0, 0], [255, 180, 0]])
    
    def __init__(self, root: str = 'data/singapore_gr', split: str = 'train', transform = None, modals = ['img'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.files = sorted(glob.glob(os.path.join(*[root, 'img', split, '*.png'])))
        # --- debug
        # self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))[:100]
        # --- split as case
        if case is not None:
            assert case in ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'], "Case name not available."
            _temp_files = [f for f in self.files if case in f]
            self.files = _temp_files
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} {case} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        rgb = str(self.files[index])
        name = rgb.split('\\')[-1]
        x1 = rgb.replace('\\img', '\\building_height')
        x2 = rgb.replace('\\img', '\\nir')
        x3 = rgb.replace('\\img', '\\osm_building')
        x4 = rgb.replace('\\img', '\\ndvi')
        lbl_path = rgb.replace('\\img', '\\semantic')

        sample = {}
        sample['img'] = torch.tensor(cv2.imread(rgb)[:, :, (2, 1, 0)].transpose(2,0,1))
        img = cv2.imread(rgb)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H, W = sample['img'].shape[1:]
        if 'building_height' in self.modals:
            sample['building_height'] = self._open_img(x1)
        if 'nir' in self.modals:
            sample['nir'] = self._open_img(x2)
        if 'osm_building' in self.modals:
            sample['osm_building'] = self._open_img(x3)  ###*255
        if 'ndvi' in self.modals:
            sample['ndvi'] = self._open_img(x4)
        label = torch.tensor(cv2.imread(lbl_path)[:, :, (2, 1, 0)].transpose(2,0,1))[0,...].unsqueeze(0)
        sample['mask'] = label
        
        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]

        return sample, label,img, name

    def _open_img(self, file):
        img = torch.tensor(cv2.imread(file)[:, :, (2, 1, 0)].transpose(2,0,1))
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)


if __name__ == '__main__':
    cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    traintransform = get_train_augmentation_adapt((1024, 1024), seg_fill=255)
    for case in cases:
        trainset = Mydata(transform=traintransform, split='val', case=case)
        trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

        for i, (sample, lbl) in enumerate(trainloader):
            print(torch.unique(lbl))