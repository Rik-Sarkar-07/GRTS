from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomHorizontalFlip,
    ColorJitter, ToTensor
)
from timm.data import RandAugment

def get_transforms(args, train=True):
    transforms = [
        RandomResizedCrop(args.input_size),
        RandomHorizontalFlip(),
    ]

    if train:
        if args.colorjitter:
            transforms.append(ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ))
        if args.randaug:
            transforms.append(RandAugment(num_ops=2, magnitude=9))

    transforms += [
        ToTensor(),   # <-- ONLY convert to [0,1]
    ]

    return Compose(transforms)
