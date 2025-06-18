import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.utils import color_map
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale


class SUNRGBD(data.Dataset):

    def __init__(self, cfg, mode='train', do_aug=True):
        assert mode in ['train', 'test']

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449], [0.226]),
        ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']
        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            Resize(crop_size),
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])
        self.val_resize = Resize(crop_size)
        self.class_weight = np.array([4.2026, 4.8137, 5.8542, 23.2349, 20.5158, 10.1502, 23.4488, 13.8937,
                                      24.4779, 23.0975, 40.1474, 39.4138, 37.1409, 42.8543, 27.0614, 43.4786,
                                      35.6323, 41.7237, 40.2385, 40.0898, 49.8679, 44.2955, 36.1612, 43.6507,
                                      45.2532, 46.5176, 44.2880, 46.7719, 49.9860, 41.0655, 40.0400, 48.2056,
                                      48.5913, 44.9882, 42.8823, 45.5003, 46.3329, 46.1943])  # enet

        self.mode = mode
        self.do_aug = do_aug

        self.train_ids = list(range(5384))
        self.test_ids = list(range(5050))

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_ids)
        else:
            return len(self.test_ids)

    def __getitem__(self, index):
        if self.mode == 'train':
            image_index = self.train_ids[index]
        else:
            image_index = self.test_ids[index]

        image_path = f'{self.mode}/image/{image_index}.jpg'
        depth_path = f'{self.mode}/depth/{image_index}.png'
        label_path = f'{self.mode}/label/{image_index}.png'
        image = Image.open(os.path.join(self.root, image_path))  # RGB 0~255
        depth = Image.open(os.path.join(self.root, depth_path))  # 1 channel -> 3
        label = Image.open(os.path.join(self.root, label_path))  # 1 channel 0~37

        sample = {
            'image': image,
            'depth': depth,
            'label': label,
        }

        if self.mode == 'train' and self.do_aug:
            sample = self.aug(sample)
        else:
            sample = self.val_resize(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()

        sample['label_path'] = label_path.strip().split('/')[-1] 
        return sample

    @property
    def cmap(self):
        return color_map(N=self.n_classes)
