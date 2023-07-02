import os
import torch
import torch.utils.data as data
from PIL import Image
from glob import glob
# from core.visualize import save_img


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class StaticImage(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.img_list = list()

        dataset_list = list()
        # lines = ['COCO', 'MSRA10K', 'PASCAL']
        lines = ['COCO']
        for line in lines:
            dataset_name = line.strip()
            img_dir = os.path.join(root, 'JPEGImages', dataset_name)

            img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
                
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def get_clip(self, idx):
        img_path = self.img_list[idx]
        seq = pil_loader(img_path)

        return seq

    def __getitem__(self, idx):
        img = self.get_clip(idx)
        frames_per_clip = 1

        video1_frames, video2_frames = [], []
        video1_coords, video2_coords = [], []
        frame = img
        img1, coord1 = self.transform[0](frame)
        img2, coord2 = self.transform[1](frame)
        video1_frames.append(img1)
        video1_coords.append(coord1)
        video2_frames.append(img2)
        video2_coords.append(coord2)

        video1 = torch.stack(video1_frames)  # [T, 3, 256, 256]
        video2 = torch.stack(video2_frames)  # [T, 3, 256, 256]
        coord1 = torch.stack(video1_coords)  # [T, 4]
        coord2 = torch.stack(video2_coords)  # [T, 4]

        return (video1, video2), (coord1, coord2)