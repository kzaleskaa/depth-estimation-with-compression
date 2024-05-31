import os
from typing import Tuple

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NYUDataset(Dataset):
    def __init__(
        self, file_name: str, data_dir: str, transform=None, target_transform=None
    ) -> None:
        self.file_name = file_name
        self.data_dir = data_dir
        self.df = self.load_df()
        self.transform = transform
        self.target_transform = target_transform

    def load_df(self) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, self.file_name)
        return pd.read_csv(file_path, names=["img", "depth"], header=None)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.df.iloc[idx, 0]
        mask_path = self.df.iloc[idx, 1]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # img = np.array(img)
        mask =  np.asarray(mask, np.float64)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask).squeeze(0)

        img = img.float() / 255.0
        mask = mask.float() / 255.0

        return img, mask
