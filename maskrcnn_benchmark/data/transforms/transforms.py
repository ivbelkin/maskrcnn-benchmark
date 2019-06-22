# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
from shapely import geometry, affinity
from shapely.geometry import Polygon


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size, Image.LANCZOS)
        if target is None:
            return image, None
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class IcevisionCrop(object):
    def __init__(self, p=0):
        self.p = p

    def __call__(self, image, target):
        size = image.size
        try:
            if np.random.uniform() < self.p:
                keep = []
                tries = 0
                while len(keep) == 0:
                    crop_box = self._gen_crop_points(size)
                    crop_size = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
                    crop_polygon = geometry.box(*crop_box)
                    keep = self._get_keep_idxs(target, crop_polygon)
                    tries += 1
                    if tries >= 10:
                        break

                if len(keep) > 0:
                    xoff, yoff = crop_box[:2]
                    boxes, masks, labels = [], [], []
                    for i in keep:
                        box = target.bbox[i]
                        polygon = geometry.box(*box)
                        polygon = crop_polygon.intersection(polygon)
                        polygon = affinity.translate(polygon, -xoff, -yoff)
                        box = polygon.bounds
                        boxes.append(box)

                        mask = target.get_field("masks")[i].instances.polygons[0].polygons
                        assert len(mask) == 1
                        polygon = Polygon(mask[0].view(-1, 2))
                        polygon = crop_polygon.intersection(polygon)
                        polygon = affinity.translate(polygon, -xoff, -yoff)
                        mask = [coord for point in polygon.exterior.coords for coord in point]
                        masks.append([mask])

                        labels.append(target.get_field("labels")[i])

                    boxes = torch.tensor(boxes, dtype=torch.float32)
                    labels = torch.tensor(labels)

                    masks = SegmentationMask(masks, crop_size, mode='poly')

                    target = BoxList(boxes, crop_size, mode="xyxy")
                    target.add_field("labels", labels)
                    target.add_field("masks", masks)

                    image = image.crop(crop_box).resize(size, resample=Image.LANCZOS)
                    target = target.resize(size)
        except:
            print("Oops!")

        return image, target

    @staticmethod
    def _gen_crop_points(image_size):
        bottom = np.random.randint(1100, 1300)
        h = np.random.randint(600, bottom)
        w = int(h * image_size[0] / image_size[1])
        max_delta_w = int(w * 0.05)
        w += np.random.randint(-max_delta_w, max_delta_w + 1)
        left = np.random.randint(0, image_size[0] - w)
        return left, bottom - h, left + w, bottom

    @staticmethod
    def _get_keep_idxs(target, crop_polygon):
        keep = []
        for i, box in enumerate(target.bbox):
            polygon = geometry.box(*box)
            polygon = crop_polygon.intersection(polygon)
            box = polygon.bounds
            if len(box) > 0:
                m = min(np.abs(box[0] - box[2]), np.abs(box[1] - box[3]))
                if m >= 24:
                    keep.append(i)
        return keep
