import math

import torch
from torch import nn
from torchvision.transforms import functional as F
import random


def _resize_image(image, self_min_size, self_max_size):
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))
    max_size = float(torch.max(im_shape))
    scale_factor = self_min_size / min_size
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size
    image = torch.nn.functional.interpolate(image[None], scale_factor=scale_factor, mode="bilinear",
                                            recompute_scale_factor=True, align_corners=False)[0]
    return image


def resize_boxes(boxes, original_szie, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_szie)]
    ratios_height, ratios_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size  # 指定图像的最小边长范围
        self.max_size = max_size  # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std  # 指定图像在标准化处理中的方差

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self, image, targets):
        h, w = image.shape[-2:]
        size = float(self.min_size[-1])
        image = _resize_image(image, size, self.max_size)
        if targets is None:
            return image, targets
        bbox = targets["boxes"]
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        targets["boxes"] = bbox
        return image, targets

    def max_by_axis(self, the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):  # size_divisible:将长和宽调整到该数的整数倍
        max_size = self.max_by_axis([list(img.shape) for img in images])  # max_size:[max_channel,max_width,max_height]
        stride = float(size_divisible)
        max_size[1] = int(math.ceil(float(max_size[1] / stride)) * stride)  # width和height向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2] / stride)) * stride)
        batch_shape = [len(images)] + max_size  # [8,3,224,224]
        batched_images = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_images):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        return batched_images

    def forward(self, images, targets=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            assert image.dim() == 3

            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None:
                targets[i] = target_index

        # resize之后的尺寸
        image_sizes = [img.shape[-2:] for img in images]
        image_sizes_list = []
        images = self.batch_images(images)
        for img_size in image_sizes:
            assert len(img_size) == 2
            image_sizes_list.append((img_size[0], img_size[1]))  # 这个保存的是resize以后,batch以前的尺寸

        images_list = ImageList(images, image_sizes_list)
        return images_list, targets


class ImageList(object):
    def __init__(self, tensors, image_sizes):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


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
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target
