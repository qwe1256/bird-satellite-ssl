import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
import random
from PIL import Image

class PairedImageFolderDataset(ImageFolder):
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
            bird_img = self.transform(bird_img)
            satellite_img = self.transform(satellite_img)

        return bird_img, satellite_img, target

class ImageOnlyTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return self.transform(img)


def main():
    b_path = r'C:\Users\Hugo\Desktop\bird'
    s_path = r'C:\Users\Hugo\Desktop\satellite'
    # Define your transformations
    
    
    augmentation = [
        transforms.Resize(224*2),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ]
    # Define the dataset
    dataset = PairedImageFolderDataset(bird_root=b_path,
                                    satellite_root=s_path,
                                    transform=ImageOnlyTransform(transforms.Compose(augmentation)))

    # Define the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for b, s, t in dataloader:
        print(s.shape)
        break
    
if __name__ == "__main__":
    main()