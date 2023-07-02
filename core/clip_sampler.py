import torch
from torch.utils.data import Sampler
from typing import Iterator


class RandomClipSamplerFrame(Sampler):

    def __init__(self, resampling_idxs, max_clips_per_video) -> None:
        self.resampling_idxs = resampling_idxs
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self) -> Iterator[int]:
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, randomly
        for c in self.resampling_idxs:
            length = len(c)
            size = min(length, self.max_clips_per_video)
            sampled = torch.randperm(length)[:size] + s
            s += length
            idxs.append(sampled)
        idxs_ = torch.cat(idxs)
        # shuffle all clips randomly
        perm = torch.randperm(len(idxs_))
        return iter(idxs_[perm].tolist())

    def __len__(self) -> int:
        return sum(min(len(c), self.max_clips_per_video) for c in self.resampling_idxs)
