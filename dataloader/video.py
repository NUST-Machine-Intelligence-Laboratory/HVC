import os
import csv
import bisect
import torch
import torch.utils.data as data
from PIL import Image
# from core.visualize import save_img

def unfold(tensor, size, step, dilation=1):
    """
    similar to tensor.unfold, but with the dilation
    and specialized for 1d tensors

    Returns all consecutive windows of `size` elements, with
    `step` between windows. The distance between each element
    in a window is given by `dilation`.
    """
    assert tensor.dim() == 1
    o_stride = tensor.stride(0)
    numel = tensor.numel()
    new_stride = (step * o_stride, dilation * o_stride)
    new_size = ((numel - (dilation * (size - 1) + 1)) // step + 1, size)
    if new_size[0] < 1:
        new_size = (0, size)
    return torch.as_strided(tensor, new_size, new_stride)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class YoutubeVOS(data.Dataset):
    def __init__(self, root, frames_per_clip=1, step_between_clips=1, frame_rate=30,
                 transform=None):
        # csv_path = os.path.join(root, 'JPEGImages')
        # if not root.endswith(".csv"):
        #     print("Did not detect .csv file, scan dir {} and generate ytvos.csv".format(csv_path))
        #     ld = os.listdir(csv_path)
        #     with open(os.path.join(root, 'ytvos.csv'), 'w') as f:
        #         filewriter = csv.writer(f)
        #         for l in ld:
        #             files = os.listdir(os.path.join(csv_path, l))
        #             n = len(files)
        #             files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        #             init = int(files[0].split('.')[0])
        #             filewriter.writerow([l, init, n])
        #     csv_path = os.path.join(root, 'ytvos.csv')
        csv_path = os.path.join(root, 'ytvos.csv')
        lines = open(csv_path).readlines()
        video_names = [line.split(',')[0].strip() for line in lines]
        start_frames = [int(line.split(',')[1].strip()) for line in lines]
        video_lengths = [int(line.split(',')[2].strip()) for line in lines]

        self.video_paths = [os.path.join(root, 'JPEGImages', name) for name in video_names]

        self.resampling_idxs = []
        for start, length in zip(start_frames, video_lengths):
            idxs = torch.arange(length)
            dilation = round(30 / float(frame_rate))
            idxs = unfold(idxs, frames_per_clip, step_between_clips, dilation=dilation)
            idxs = idxs + start
            self.resampling_idxs.append(idxs)
        clip_lengths = torch.as_tensor([len(v) for v in self.resampling_idxs])
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

        self.frames_per_clip = frames_per_clip
        self.transform = transform

    def __len__(self):
        return self.cumulative_sizes[-1]

    def num_clips(self):
        """
        Number of subclips that are available in the video list.
        """
        return self.cumulative_sizes[-1]

    def get_clip_location(self, idx):
        """
        Converts a flattened representation of the indices into a video_idx, clip_idx
        representation.
        """
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    def get_clip(self, idx):
        if idx >= self.num_clips():
            raise IndexError(
                "Index {} out of range "
                "({} number of clips)".format(idx, self.num_clips())
            )

        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        resampling_idx = self.resampling_idxs[video_idx][clip_idx]
        resampling_idx = resampling_idx.numpy()
        seq = [pil_loader(os.path.join(video_path, '{:05d}.jpg'.format(i))) for i in resampling_idx]

        return seq

    def __getitem__(self, idx):
        video = self.get_clip(idx)
        frames_per_clip = len(video)

        video1_frames, video2_frames = [], []
        video1_coords, video2_coords = [], []
        for t in range(frames_per_clip):
            frame = video[t]
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