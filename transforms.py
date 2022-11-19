import torch
from torch import nn
from torchvision.transforms import functional as F
import random


class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size  # 指定图像的最小边长范围
        self.max_size = max_size  # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std  # 指定图像在标准化处理中的方差


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self,prob = 0.5):
        self.prob = prob
    def __call__(self, image,target):
        if random.random() <self.prob:
            height ,width = image.shape[:-2]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:,[0,2]] =width - bbox[:,[2,0]]
            target["boxes"] = bbox
        return image,target
