import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class DIV2KDataset(Dataset):
    def __init__(self, lr_dir: str, hr_dir: str, hr_crop_size: int = 96):
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        if len(self.lr_paths) != len(self.hr_paths):
            raise ValueError(f"Number of LR images ({len(self.lr_paths)}) does not match HR images ({len(self.hr_paths)})")

        self.hr_crop = hr_crop_size
        self.lr_crop = hr_crop_size // 2

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, idx: int):
        hr_image = Image.open(self.hr_paths[idx]).convert("RGB")

        # Step 1: Crop only HR image
        hr_image = transforms.CenterCrop(self.hr_crop)(hr_image)

        # Step 2: Resize HR to LR
        lr_image = hr_image.resize((self.lr_crop, self.lr_crop), Image.BICUBIC)

        # Step 3: Convert both to tensors
        hr = self.to_tensor(hr_image)
        lr = self.to_tensor(lr_image)

        return lr, hr


def get_dataloaders(
    train_lr_dir: str,
    train_hr_dir: str,
    valid_lr_dir: str,
    valid_hr_dir: str,
    hr_crop_size: int = 96,
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool = True
):
    """
    Creates DataLoader objects for training and validation sets.

    Returns:
        train_loader, valid_loader (DataLoader, DataLoader)
    """
    train_dataset = DIV2KDataset(train_lr_dir, train_hr_dir, hr_crop_size)
    valid_dataset = DIV2KDataset(valid_lr_dir, valid_hr_dir, hr_crop_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, valid_loader
