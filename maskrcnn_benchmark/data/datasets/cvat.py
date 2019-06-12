import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from icevision.cvat import CvatDataset


class CVATDataset(Dataset):

    def __init__(
        self,
        annot_xml,
        image_folder,
        skip_smaller_than=100,
        keep_in_ram=False,
        transforms=None
    ):
        super().__init__()
        self.annot_xml = annot_xml
        self.image_folder = image_folder
        self.skip_smaller_than = skip_smaller_than
        self.keep_in_ram = keep_in_ram
        self.transforms = transforms

        self._ds = CvatDataset()
        self._ds.load(annot_xml)
        self._index_to_image_id = self._ds.get_image_ids()

        self.class_to_ind = {label: i + 1 for i, label in enumerate(self._ds.get_labels())}
        self.class_to_ind["__background__"] = 0

        self.data = self.prepare_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.keep_in_ram and "img" in self.data[index]:
            img = self.data[index]["img"]
        else:
            filename = self.data[index]["filename"]
            path = os.path.join(self.image_folder, filename)
            img = Image.open(path).convert("RGB")

        if self.keep_in_ram:
            self.data[index]["img"] = img

        target = self.data[index]["target"]
        target = target.clip_to_image(remove_empty=True)

        target.add_field("orig_size", torch.tensor(img.size))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_img_info(self, index):
        return self.data[index]["size"]

    def prepare_data(self):
        annotations = []

        for image_id in self._index_to_image_id:
            filename = os.path.basename(self._ds.get_name(image_id))

            size = self._ds.get_size(image_id)
            size = (size["width"], size["height"])

            areas, boxes, labels = [], [], []
            for box in self._ds.get_boxes(image_id):
                area = np.abs((box["xtl"] - box["xbr"]) * (box["ytl"] - box["ybr"]))
                areas.append(area)
                if area >= self.skip_smaller_than:
                    boxes.append([box["xtl"], box["ytl"], box["xbr"], box["ybr"]])
                    labels.append(self.class_to_ind[box["label"]])

            if len(boxes) == 0:
                continue

            polygons = self._ds.get_polygons(image_id)
            if polygons:
                masks = []
                for area, polygon in zip(areas, polygons):
                    if area >= self.skip_smaller_than:
                        masks.append([[coord for point in polygon["points"] for coord in point]])
            else:
                masks = [
                    CVATDataset.dummy_mask(box) for area, box in zip(areas, boxes) if area > self.skip_smaller_than
                ]

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels)

            masks = SegmentationMask(masks, size, mode='poly')

            target = BoxList(boxes, size, mode="xyxy")
            target.add_field("labels", labels)
            target.add_field("masks", masks)

            annotations.append({"filename": filename, "target": target, "size": self._ds.get_size(image_id)})

        return annotations

    @staticmethod
    def dummy_mask(box):
        xtl, ytl, xbr, ybr = box
        return [[xtl, ytl, xbr, ytl, xbr, ybr, xtl, ybr]]
