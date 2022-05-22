
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2 as cv

class CoinImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_lables = pd.read_csv(annotations_file)
        self.transform = transform
        self.traget_transform = target_transform

    def __len__(self):
        return len(self.img_lables)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_lables.iloc[idx, 0])

        image = cv.imread(img_path + ".jpg")

        label = self.img_lables.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "label": label}

        return sample