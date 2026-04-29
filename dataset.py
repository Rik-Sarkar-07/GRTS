from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, loader=self.custom_loader)

    def custom_loader(self, path):
        if path.lower().endswith(('.tif', '.jpg')):
            return Image.open(path).convert('RGB')
        raise ValueError(f"Unsupported file format: {path}")

def get_dataset(data_dir, dataset_name, train_transform, val_transform):
    train_root = os.path.join(data_dir, 'train')
    val_root = os.path.join(data_dir, 'val')

    if not os.path.exists(train_root) or not os.path.exists(val_root):
        raise ValueError("train and val folders not found in data_dir")

    train_dataset = CustomImageFolder(train_root, transform=train_transform)
    val_dataset = CustomImageFolder(val_root, transform=val_transform)

    num_classes = len(train_dataset.classes)

    return train_dataset, val_dataset, num_classes