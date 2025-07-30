import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class MUCADDataset(Dataset):
    """
    Dataset class for the MUCAD 'captures' folder structure.
    """
    def __init__(self, dataset_root_path, img_height, img_width, msi_channels, subset_indices=None, mode="train"):
        self.dataset_root_path = dataset_root_path
        self.img_height = img_height
        self.img_width = img_width
        self.msi_channels = msi_channels
        self.mode = mode
        self.all_data_items_paths = []

        if subset_indices is None:
            if not os.path.exists(dataset_root_path):
                raise FileNotFoundError(f"CRITICAL ERROR: Dataset path does not exist: {dataset_root_path}")
            rgb_files = sorted(glob.glob(os.path.join(dataset_root_path, "*_vis.png")))
            for rgb_path in rgb_files:
                base_name = rgb_path.replace("_vis.png", "")
                all_bands = sorted(glob.glob(f"{base_name}_*.png"))
                msi_paths = [p for p in all_bands if p != rgb_path]
                if len(msi_paths) == self.msi_channels:
                    self.all_data_items_paths.append((rgb_path, msi_paths))
            self.current_data_items = self.all_data_items_paths
        else:
            self.current_data_items = subset_indices

        self.resize_transform = transforms.Resize((self.img_height, self.img_width))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.current_data_items)

    def __getitem__(self, index):
        rgb_path, msi_band_paths = self.current_data_items[index]
        try:
            rgb_image_pil = Image.open(rgb_path).convert('RGB')
            rgb_tensor = self.to_tensor(self.resize_transform(rgb_image_pil))
            msi_bands_tensors = [self.to_tensor(self.resize_transform(Image.open(p).convert('L'))) for p in msi_band_paths]
            msi_tensor = torch.cat(msi_bands_tensors, dim=0)
            if self.mode == "train":
                if random.random() > 0.5:
                    rgb_tensor, msi_tensor = TF.hflip(rgb_tensor), TF.hflip(msi_tensor)
                if random.random() > 0.5:
                    angle = random.choice([90, 180, 270])
                    rgb_tensor, msi_tensor = TF.rotate(rgb_tensor, angle), TF.rotate(msi_tensor, angle)
            return rgb_tensor, msi_tensor
        except Exception as e:
            print(f"Error loading item {index} (RGB path: {rgb_path}): {e}")
            return torch.rand(3, self.img_height, self.img_width), torch.rand(self.msi_channels, self.img_height, self.img_width)