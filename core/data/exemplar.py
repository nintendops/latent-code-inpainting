import os
import numpy as np
import albumentations
import glob
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from core.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from PIL import Image

class ImageFolder(Dataset):
    def __init__(self, size=None, dataroot="", multiplier=5000, onehot_segmentation=False, 
                 crop_size=None, force_no_crop=False, given_files=None, multiscale_factor=1.0, extension='jpg', split='train'):
        self.ext = extension
        self.split = split # self.get_split()
        self.size = size
        self.multiplier = multiplier
        self.ms_factor = multiscale_factor
        if crop_size is None:
            self.crop_size = size
        else:
            self.crop_size = crop_size
        self.dataroot = dataroot
        self.initialize_paths()
        self.initialize_processor(force_no_crop)

    def __len__(self):
        return len(self.labels["image_ids"])

    def initialize_paths(self):
        # file paths without extensions
        ids = [f[:-4].split('/')[-1] for f in glob.glob(os.path.join(self.dataroot, f"*.{self.ext}"))] # self.json_data["images"]     
        ids = ids*self.multiplier
        self.labels = {"image_ids": ids}
        self.img_id_to_filepath = dict()
        for iid in tqdm(ids, desc='ImgToPath'):
            self.img_id_to_filepath[iid] =  os.path.join(self.dataroot, iid+f'.{self.ext}')

    def initialize_processor(self, force_no_crop=False):
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)

        if self.split=="validation":
            self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            self.hflipper = albumentations.HorizontalFlip(p=0.0)
        else:
            self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
            self.hflipper = albumentations.HorizontalFlip(p=0.5)

        self.preprocessor = albumentations.Compose(
            [self.rescaler, self.cropper, self.hflipper])
        if force_no_crop:
            self.rescaler = albumentations.Resize(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose(
                [self.rescaler])
        if self.ms_factor < 1.0:
            self.rescaler_2 = albumentations.Resize(height=int(self.ms_factor*self.size), width=int(self.ms_factor*self.size))

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        processed = self.preprocessor(image=image)
        image = processed["image"]
        if self.ms_factor < 1.0:
            image_rescaled = self.rescaler_2(image=image)['image']
        image = (image / 127.5 - 1.0).astype(np.float32)
        if self.ms_factor < 1.0:
            image_rescaled = (image_rescaled / 127.5 - 1.0).astype(np.float32)
        else:
            image_rescaled = image
        return image, image_rescaled

    def __getitem__(self, i):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        image, image_rescaled = self.preprocess_image(img_path)
        example = {"image": image,
                   "img_path": img_path,
                   "filename_": img_path.split(os.sep)[-1]
                    }
        return example
