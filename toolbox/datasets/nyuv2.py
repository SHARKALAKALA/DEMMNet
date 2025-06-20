import os
from PIL import Image
from scipy.io import loadmat
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale
from torchvision import transforms


class NYUv2(data.Dataset):

    def __init__(self, cfg, random_state=3, mode='train', ):
        assert mode in ['train', 'test']

        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),  # p=0.5
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])
        self.val_size = Resize(crop_size)
        self.mode = mode
        self.class_weight = np.array([4.01302219, 5.17995767, 12.47921102, 13.79726557, 18.47574439, 19.97749822,
                                      21.10995738, 25.86733191, 27.50483598, 27.35425244, 25.12185149, 27.04617447,
                                      30.0332327, 29.30994935, 34.72009825, 33.66136128, 34.28715586, 32.69376342,
                                      33.71574286, 37.0865665, 39.70731054, 38.60681717, 36.37894266, 40.12142316,
                                      39.71753044, 39.27177794, 43.44761984, 42.96761184, 43.98874667, 43.43148409,
                                      43.29897719, 45.88895515, 44.31838311, 44.18898992, 42.93723439, 44.61617778,
                                      47.12778303, 46.21331253, 27.69259756, 25.89111664, 15.65148615, ])

        SPLITS_FILEPATH = './toolbox/datasets/nyudv2_splits.mat'
        splits = loadmat(SPLITS_FILEPATH)
        self.train_ids = splits['trainNdxs'][:, 0]
        self.test_ids = splits['testNdxs'][:, 0]

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

        image_path = f'images/{image_index}.jpg'
        depth_path = f'depth/{image_index}.png'
        label_path = f'label/{image_index}.png'

        image = Image.open(os.path.join(self.root, image_path))
        depth = Image.open(os.path.join(self.root, depth_path))
        label = Image.open(os.path.join(self.root, label_path))

        sample = {
            'image': image,
            'depth': depth,
            'label': label,
        }

        if self.mode == 'train':
            sample = self.aug(sample)
        else:
            sample = self.val_size(sample)
        self.class_weight = np.array([3.3855, 5.6249, 12.0095, 13.9648, 21.0021, 20.5423, 25.7143, 27.3796,
                                      28.7891, 27.1876, 28.5285, 28.1800, 32.1900, 31.0330, 35.3340, 32.6370,
                                      36.1279, 36.2773, 37.1518, 35.4062, 40.4284, 37.5624, 33.7325, 40.8962,
                                      40.9146, 40.2932, 43.6822, 44.7070, 43.7316, 42.5433, 45.0084, 44.8754,
                                      45.2884, 44.5937, 45.4827, 45.2658, 46.5859, 46.4504, 26.3095, 27.8991,
                                      16.6582])  # enet
        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [(0, 0, 0),
                (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (64, 0, 0), (192, 0, 0), (64, 128, 0),
                (192, 128, 0), (64, 0, 128), (192, 0, 128),
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128),
                (0, 192, 128), (128, 192, 128), (64, 64, 0), (192, 64, 0),
                (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128),
                (64, 192, 128), (192, 192, 128), (0, 0, 64), (128, 0, 64),
                (0, 128, 64), (128, 128, 64), (0, 0, 192), (128, 0, 192),
                (0, 128, 192), (128, 128, 192), (64, 0, 64)]  # 41