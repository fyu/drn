import numbers
import random

import numpy as np
from PIL import Image, ImageOps
import torch


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, label=None, *args):
        assert label is None or img.size == label.size

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return (img, label, *args)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        results = [img.crop((x1, y1, x1 + tw, y1 + th))]
        if label is not None:
            results.append(label.crop((x1, y1, x1 + tw, y1 + th)))
        results.extend(args)
        return results


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, image, label):
        if random.random() < 0.5:
            results = [image.transpose(Image.FLIP_LEFT_RIGHT),
                       label.transpose(Image.FLIP_LEFT_RIGHT)]
        else:
            results = [image, label]
        return results


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, image, label=None):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        if label is None:
            return image,
        else:
            return image, label


class Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
               isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, image, label=None, *args):
        if label is not None:
            label = ImageOps.expand(label, border=self.padding, fill=255)
        if self.fill == -1:
            image = np.asarray(image)
            image = cv2.copyMakeBorder(image, self.padding, self.padding,
                                       self.padding, self.padding,
                                       cv2.BORDER_REFLECT_101)
            image = Image.fromarray(image)
            return (image, label, *args)
        else:
            return ImageOps.expand(image, border=self.padding, fill=self.fill), label


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, label=None):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255)
        if label is None:
            return img,
        else:
            return img, torch.LongTensor(np.array(label, dtype=np.int))


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
