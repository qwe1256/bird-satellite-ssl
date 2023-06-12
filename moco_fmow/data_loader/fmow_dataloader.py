import pandas as pd
import numpy as np
import warnings
import random
from glob import glob

from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class Bird_Satellite_Dataset(ImageFolder):
    def __init__(self, bird_root, satellite_root, transform=None):
        super().__init__(bird_root, transform)
        self.satellite_root = Path(satellite_root)

    def get_random_satellite_path(self, class_name):
        class_path = self.satellite_root / class_name
        image_files = [f for f in class_path.iterdir() if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        return str(random.choice(image_files))

    def __getitem__(self, index):
        # Get bird image and its label
        bird_path, target = self.samples[index]
        bird_img = Image.open(bird_path)
        
        # Get random satellite image from same class
        satellite_path = self.get_random_satellite_path(self.classes[target])
        satellite_img = Image.open(satellite_path)

        if self.transform is not None:
            bird_tensor = self.transform(bird_img)
            satellite_tensor = self.transform(satellite_img)

        return [bird_tensor, satellite_tensor], target
