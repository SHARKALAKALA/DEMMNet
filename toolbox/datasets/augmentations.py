from __future__ import division
import sys
import random

import cv2
import numpy as np
from PIL import Image
import math
try:
    import accimage
except ImportError:
    accimage = None
import numbers
import collections

import torchvision.transforms.functional as F

__all__ = ["Compose",
           "Resize",
           "RandomScale", 
           "RandomCrop",
           "RandomHorizontalFlip",
           "ColorJitter",
           ]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST', 
    Image.BILINEAR: 'PIL.Image.BILINEAR', 
    Image.BICUBIC: 'PIL.Image.BICUBIC', 
    Image.LANCZOS: 'PIL.Image.LANCZOS',  
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}
_pil_interpolation_to_str = {
    F.InterpolationMode.NEAREST: 'InterpolationMode.NEAREST',
    F.InterpolationMode.BILINEAR: 'InterpolationMode.BILINEAR',
    F.InterpolationMode.BICUBIC: 'InterpolationMode.BICUBIC',
    F.InterpolationMode.LANCZOS: 'InterpolationMode.LANCZOS',
    F.InterpolationMode.HAMMING: 'InterpolationMode.HAMMING',
    F.InterpolationMode.BOX: 'InterpolationMode.BOX',
}

# 查看版本信息
if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """


    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize(object):
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, sample):
        assert 'image' in sample.keys()
        assert 'label' in sample.keys()

        sample['image'] = F.resize(sample['image'], self.size, F.InterpolationMode.BILINEAR)
        # NEAREST
        if 'depth' in sample.keys():
            sample['depth'] = F.resize(sample['depth'], self.size, F.InterpolationMode.BILINEAR)
        sample['label'] = F.resize(sample['label'], self.size, F.InterpolationMode.NEAREST)

        return sample


class Rescale:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = F.resize(image, (self.height, self.width), F.InterpolationMode.BILINEAR)
        depth = F.resize(depth, (self.height, self.width), F.InterpolationMode.NEAREST)

        sample['image'] = image
        sample['depth'] = depth

        if 'label' in sample:
            label = sample['label']
            label = F.resize(label, (self.height, self.width), F.InterpolationMode.NEAREST)
            sample['label'] = label

        return sample


def get_2dshape(shape, *, zero=True):
    if not isinstance(shape, collections.Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
    return shape


def random_crop_pad_to_shape(img, crop_pos, crop_size, pad_label_value):
    h, w = img.shape[:2]
    start_crop_h, start_crop_w = crop_pos
    assert ((start_crop_h < h) and (start_crop_h >= 0))
    assert ((start_crop_w < w) and (start_crop_w >= 0))

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    img_crop = img[start_crop_h:start_crop_h + crop_h,
               start_crop_w:start_crop_w + crop_w, ...]

    img_, margin = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT,
                                      pad_label_value)

    return img_, margin


def generate_random_crop_pos(ori_size, crop_size):
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0

    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w


def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2
    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3], border_mode, value=value)
    return img, margin
from torchvision import transforms

class RandomCrop(object):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=output_size)
        return i, j, h, w

    def __call__(self, sample):
        img = sample['image']
        if self.padding is not None:
            for key in sample.keys():
                sample[key] = F.pad(sample[key], self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            for key in sample.keys():
                sample[key] = F.pad(sample[key], (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            for key in sample.keys():
                sample[key] = F.pad(sample[key], (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(sample['image'], self.size)
        for key in sample.keys():
            sample[key] = F.crop(sample[key], i, j, h, w)

        return sample


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            for key in sample.keys():
                sample[key] = F.hflip(sample[key])

        return sample


class RandomScale(object):
    def __init__(self, scale):
        assert isinstance(scale, Iterable) and len(scale) == 2
        assert 0 < scale[0] <= scale[1]
        self.scale = scale

    def __call__(self, sample):
        assert 'image' in sample.keys()
        assert 'label' in sample.keys()

        w, h = sample['image'].size
        scale = random.uniform(self.scale[0], self.scale[1])
        size = (int(round(h * scale)), int(round(w * scale)))

        # BILINEAR
        sample['image'] = F.resize(sample['image'], size, F.InterpolationMode.BILINEAR)

        # NEAREST
        if 'depth' in sample.keys():
            sample['depth'] = F.resize(sample['depth'], size, F.InterpolationMode.NEAREST)
        sample['label'] = F.resize(sample['label'], size, F.InterpolationMode.NEAREST)

        return sample


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):

        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        assert 'image' in sample.keys()
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        sample['image'] = transform(sample['image'])
        return sample
