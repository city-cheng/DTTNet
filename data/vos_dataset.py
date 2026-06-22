import os
from os import path, replace

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from utils.transforms import im_normalization, im_mean
from utils.random import reseed
from utils.data_utils import sample_with_jump


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """

    def __init__(
        self,
        config:dict,
        max_jump,
        finetune=False,
    ):
        self.im_root = config['visha']['im_root']
        self.gt_root = config['visha']['gt_root']
        self.max_jump = max_jump
        self.num_frames = config['num_frames']
        self.scale = config['scale']

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < self.num_frames:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print(
            "%d out of %d videos accepted in %s."
            % (len(self.videos), len(vid_list), self.im_root)
        )

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.01, 0.01, 0.01, 0),
            ]
        )

        self.pair_im_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=0 if finetune else 15,
                    shear=0 if finetune else 10,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=im_mean,
                ),
            ]
        )

        self.pair_gt_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=0 if finetune else 15,
                    shear=0 if finetune else 10,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                ),
            ]
        )

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.1, 0.03, 0.03, 0),
                transforms.RandomGrayscale(0.05),
            ]
        )

        
      

        self.all_im_dual_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    (self.scale, self.scale),
                    scale=(0.36, 1.00),
                    interpolation=InterpolationMode.BILINEAR,
                ),
            ]
        )

        self.all_gt_dual_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    (self.scale, self.scale),
                    scale=(0.36, 1.00),
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )

        # Final transform without randomness
        self.final_im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                im_normalization,
            ]
        )
        self.final_gt_transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info["name"] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        info["frames"] = []  # Appended with actual frames

        num_frames = self.num_frames
        length = len(frames)
        this_max_jump = min(len(frames), self.max_jump)

        # iterative sampling
        clips_idx = sample_with_jump(num_frames,length,this_max_jump,mode='clip')

        sequence_seed = np.random.randint(2147483647)
        clip_images = []
        clip_masks = []
        for frames_idx in clips_idx:
            images = []
            masks = []
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + ".jpg"
                png_name = frames[f_idx][:-4] + ".png"
                info["frames"].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert("RGB")
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert("L")
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt =  self.final_gt_transform(this_gt)
                # print(f'this_gt shape:{this_gt.shape}')

                images.append(this_im)
                masks.append(this_gt)
            images = torch.stack(images, 0) # t,3,h,w
            masks = np.stack(masks, 0)      # t,1,h,w
            clip_images.append(images)
            clip_masks.append(masks)
        clip_images = torch.stack(clip_images, 0)  #clip,t,3,h,w
        clip_masks = np.stack(clip_masks, 0)       #clip,t,1,h,w

        # # 1 if object exist, 0 otherwise
        # selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        # selector = torch.FloatTensor(selector)

        data = {
            "images": clip_images,
            "masks": clip_masks,
            "info": info,
        }

        return data

    def __len__(self):
        return len(self.videos)
