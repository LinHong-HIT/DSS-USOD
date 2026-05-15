import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class USODDataset(Dataset):
    """
    train / val:
        return image, mask, edge
    test:
        return image, img_name

    dataset structure:
        root/
          train/
            RGB/
            GT/
            Boundary/
          val/
            RGB/
            GT/
            Boundary/
          test/
            RGB/
            GT/
            Boundary/
    Boundary image name:
        00001_edge.png
    """

    def __init__(self, root, mode='train', size=352):
        self.mode = mode
        self.size = size

        if mode == 'train':
            split_root = os.path.join(root, 'train')
        elif mode == 'val':
            split_root = os.path.join(root, 'val')
        elif mode == 'test':
            split_root = os.path.join(root, 'test')
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.rgb_root = os.path.join(split_root, 'RGB')
        self.gt_root = os.path.join(split_root, 'GT')
        self.edge_root = os.path.join(split_root, 'Boundary')

        self.image_list = sorted([
            f for f in os.listdir(self.rgb_root)
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
        ])

        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ], additional_targets={'edge': 'mask'})

        elif mode == 'val':
            self.transform = A.Compose([
                A.Resize(size, size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ], additional_targets={'edge': 'mask'})

        else:
            # test only load RGB
            self.transform = A.Compose([
                A.Resize(size, size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_list)

    def _load_binary_mask(self, folder, basename):
        possible_exts = ['.png', '.jpg', '.jpeg', '.bmp']
        arr = None

        for ext in possible_exts:
            p = os.path.join(folder, basename + ext)
            if os.path.exists(p):
                arr = cv2.imread(p, 0)
                if arr is None:
                    raise FileNotFoundError(f"Read error: {p}")
                break

        if arr is None:
            raise FileNotFoundError(f"Mask not found for {basename} in {folder}")

        arr = arr.astype(np.float32) / 255.0
        arr = (arr > 0.5).astype(np.float32)
        return arr

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        basename = os.path.splitext(img_name)[0]

        rgb_path = os.path.join(self.rgb_root, img_name)
        image = cv2.imread(rgb_path)
        if image is None:
            raise FileNotFoundError(f"Image read error: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # test mode: return image and image name
        if self.mode == 'test':
            augmented = self.transform(image=image)
            image = augmented['image']
            return image, img_name

        # train / val: load mask and edge
        mask = self._load_binary_mask(self.gt_root, basename)
        edge = self._load_binary_mask(self.edge_root, basename + "_edge")

        augmented = self.transform(image=image, mask=mask, edge=edge)

        image = augmented['image']
        mask = augmented['mask'].unsqueeze(0).float()
        edge = augmented['edge'].unsqueeze(0).float()

        return image, mask, edge
