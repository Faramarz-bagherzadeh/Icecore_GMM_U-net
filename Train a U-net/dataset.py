import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        #self.masks = sorted(os.listdir(mask_dir), key=lambda x:float(x.split('_')[-1].split('.')[0]))
        self.masks = os.listdir(mask_dir)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # rotate_matrix = cv2.getRotationMatrix2D(center=(1400, 1400), angle=-0.3, scale=1)
        # mask = cv2.warpAffine(src=mask, M=rotate_matrix, dsize=mask.shape)
        # redusing the mask shape
        #number_of_pixels = 4
        #mask = mask[number_of_pixels:mask.shape[0] - number_of_pixels,
        #       number_of_pixels:mask.shape[1] - number_of_pixels]

        #add pading to image
        #adding_pixel = 4
        #image = cv2.copyMakeBorder(image, adding_pixel, adding_pixel, adding_pixel, adding_pixel,borderType=cv2.BORDER_CONSTANT, value=0)
        mask = mask / mask.max()
        #mask = cv2.resize(mask, (2048, 2048), interpolation=cv2.INTER_AREA)

        mask [mask> 0.5] = 1
        mask [mask <= 0.5] = 0
        mask = mask.astype('uint8')
        #image = cv2.resize(image, (2048, 2048), interpolation=cv2.INTER_AREA)

        #reshape and normalizing image
        image = image.reshape(1, image.shape[0], image.shape[1])
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask



