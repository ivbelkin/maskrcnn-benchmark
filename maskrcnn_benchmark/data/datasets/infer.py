import os
import logging

from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class InferDataset(Dataset):

    def __init__(self, image_folder, transforms):
        super().__init__()
        self.image_folder = image_folder
        self.transforms = transforms

        logger = logging.getLogger("maskrcnn_benchmark.inference")

        self.filenames = []
        self.img_info = []
        for filename in tqdm(sorted(os.listdir(image_folder))):
            try:
                path = os.path.join(image_folder, filename)
                size = Image.open(path).size
                self.filenames.append(filename)
                self.img_info.append({"width": size[0], "height": size[1]})
            except:
                logger.error("Cannot open file {} as image".format(filename))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        path = os.path.join(self.image_folder, filename)
        img = Image.open(path).convert("RGB")
        size = img.size

        if self.transforms is not None:
            img, _ = self.transforms(img, None)

        return img, (size, filename), index

    def get_img_info(self, index):
        return self.img_info[index]
