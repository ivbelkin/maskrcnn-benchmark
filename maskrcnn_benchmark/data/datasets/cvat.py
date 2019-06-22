import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from icevision.cvat import CvatDataset
from tqdm import tqdm


class CVATDataset(Dataset):

    def __init__(
        self,
        annot_xml,
        labels_file,
        image_folder,
        min_side=32,
        keep_in_ram=False,
        transforms=None
    ):
        super().__init__()
        self.annot_xml = annot_xml
        self.labels_file = labels_file
        self.image_folder = image_folder
        self.min_side = min_side
        self.keep_in_ram = keep_in_ram
        self.transforms = transforms

        self._ds = CvatDataset()
        self._ds.load(annot_xml)
        self._index_to_image_id = self._ds.get_image_ids()

        with open(labels_file, "r") as f:
            labels = [l.strip() for l in f.read().split(" ")]
        self.class_to_ind = {label: i + 1 for i, label in enumerate(labels)}
        self.class_to_ind["__background__"] = 0
        self.class_to_ind["UNK"] = len(self.class_to_ind)

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

        for image_id in tqdm(self._index_to_image_id):
            filename = os.path.basename(self._ds.get_name(image_id))

            try:
                size = self._ds.get_size(image_id)
                size = (size["width"], size["height"])
            except KeyError:
                image = Image.open(os.path.join(self.image_folder, filename))
                size = image.size

            boxes, labels, use = [], [], []
            for box in self._ds.get_boxes(image_id):
                m = min(np.abs(box["xtl"] - box["xbr"]), np.abs(box["ytl"] - box["ybr"]))
                label = box["label"]
                if label != "8" and label not in self.class_to_ind:
                    for l in self.class_to_ind:
                        if len(label) < len(l) and l.startswith(label + "."):
                            raise Exception(label)
                        if label.startswith(l + "."):
                            label = l
                            break
                if label not in self.class_to_ind:
                    label = "UNK"
                if m >= self.min_side:
                    boxes.append([box["xtl"], box["ytl"], box["xbr"], box["ybr"]])
                    labels.append(self.class_to_ind[label])
                    use.append(True)
                else:
                    use.append(False)

            if len(boxes) == 0:
                continue

            polygons = self._ds.get_polygons(image_id)
            if polygons:
                masks = []
                for flag, polygon in zip(use, polygons):
                    if flag:
                        masks.append([[coord for point in polygon["points"] for coord in point]])
            else:
                masks = [
                    CVATDataset.dummy_mask(box) for box in boxes
                ]

            self._check_consistency(boxes, masks)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels)

            masks = SegmentationMask(masks, size, mode='poly')

            target = BoxList(boxes, size, mode="xyxy")
            target.add_field("labels", labels)
            target.add_field("masks", masks)

            annotations.append({
                "filename": filename,
                "target": target,
                "size": {"width": size[0], "height": size[1]}
            })

        return annotations

    @staticmethod
    def dummy_mask(box):
        xtl, ytl, xbr, ybr = box
        return [[xtl, ytl, xbr, ytl, xbr, ybr, xtl, ybr]]

    @staticmethod
    def _check_consistency(boxes, masks):
        for box, mask in zip(boxes, masks):
            mask = mask[0]
            _box = [
                min(mask[2 * i] for i in range(len(mask) // 2)),
                min(mask[2 * i + 1] for i in range(len(mask) // 2)),
                max(mask[2 * i] for i in range(len(mask) // 2)),
                max(mask[2 * i + 1] for i in range(len(mask) // 2)),
            ]
            diff = [box[2] - box[0], box[3] - box[1], box[2] - box[0], box[3] - box[1]]
            for i in range(4):
                assert np.abs(_box[i] - box[i]) < diff[i] * 0.05, (box, _box)
