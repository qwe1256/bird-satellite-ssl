import pandas as pd
import numpy as np
import warnings
import random

from torch.utils.data.dataset import Dataset
from pathlib import Path
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class LookupTable:
    def __init__(self, dir_path):
        self.lookup_table = self.build_lookup_table(dir_path)

    def build_lookup_table(self, dir_path):
        dir_path = Path(dir_path)
        sub_class_folders = sorted(dir_path.iterdir())

        lookup_table = {}
        current_index = 0

        for sub_class_folder in sub_class_folders:
            image_files = sorted(sub_class_folder.glob('*.*'))
            num_files = len(image_files)

            for i in range(num_files):
                lookup_table[current_index] = str(image_files[i])
                current_index += 1
        return lookup_table

    def fetch_image_file(self, fetch_index):
        return self.lookup_table.get(fetch_index)


class Bird_Satellite_Dataset(Dataset):
    def __init__(self, bird_path, satellite_path, bird_satellite_pair_csv, transform=None):
        self.bird_path = Path(bird_path)
        self.satellite_path = Path(satellite_path)
        self.bird_geo_pair_csv = pd.read_csv(bird_satellite_pair_csv)
        self.transform = transform
        self.bird_image_fetcher = LookupTable(self.bird_path)


    def __len__(self):
        return self.bird_geo_pair_csv.shape[0]

    def __getitem__(self, index):
        _, idx, geo_label = self.bird_geo_pair_csv.iloc[index]
        bird_img = Image.open(self.bird_image_fetcher.fetch_image_file(idx))
        satellite_img = Image.open(list(self.satellite_path.iterdir())[(idx)])
        
        if self.transform is not None:
            bird_tensor = self.transform(bird_img)
            satellite_tensor = self.transform(satellite_img)

        return [bird_tensor, satellite_tensor], geo_label


