import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import imgaug.augmenters as iaa

class DrivingDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, augment=False):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        if augment:
            self.aug = iaa.Sequential([
                iaa.SomeOf((0,3), [
                    iaa.Affine(translate_percent={"x":(-0.05,0.05), "y":(-0.02,0.02)}, rotate=(-5,5)),
                    iaa.AdditiveGaussianNoise(scale=(0,0.02*255)),
                    iaa.Multiply((0.8,1.2)),
                    iaa.LinearContrast((0.8,1.2)),
                    iaa.Fliplr(0.0) # steering flip handled below if ever used
                ])
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['center_image_path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # crop and resize: remove sky and hood, then resize to (66,200) like NVIDIA paper
        h, w = img.shape[:2]
        top = int(h * 0.35)
        bottom = int(h * 0.9)
        img = img[top:bottom, :, :]
        img = cv2.resize(img, (200,66))
        if self.augment and self.aug is not None:
            img = self.aug(image=img)
        img = img.astype(np.float32) / 255.0
        # transpose to CHW
        img = np.transpose(img, (2,0,1))
        steering = np.float32(row['steering'])
        return torch.tensor(img), torch.tensor([steering], dtype=torch.float32)
