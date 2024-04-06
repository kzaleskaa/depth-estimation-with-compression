import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class NYUDataset(Dataset):
    def __init__(self, file_name, data_dir, transform=None, target_transform=None):
        self.df = self.load_df(file_name, data_dir)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def load_df(self, file_name, data_dir):
        file_path = os.path.join(data_dir, file_name)
        return pd.read_csv(file_path, names=["img", "depth"], header=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        mask_path = self.df.iloc[idx, 1]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask).squeeze(0)

        img = img.float() / 255.0
        mask = mask.float() / 255.0

        return img, mask