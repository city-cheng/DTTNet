import os
from os import path
from config.load import load_config

# config_path = "./config/cvsd_config.yaml"
config_path = "./config/config_eval.yaml"
config = load_config(config_path)

# TODO 设置gpu优先级
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from data.visha_dataset_video import ViSha_Dataset

from progressbar import progressbar
from utils.cal_scores import Visha_scores
from torchvision.transforms import ToPILImage
from model.shadowvl import ShadowVL

"""
Arguments loading
"""


"""
Data preparation
"""

out = None

# gpu = int(config['gpus'])

# meta_dataset = VishaTestDataset(config)
meta_dataset = ViSha_Dataset("test", config)

torch.autograd.set_grad_enabled(False)

# Set up loader
# meta_loader = meta_dataset.get_datasets()
meta_loader = DataLoader(meta_dataset, batch_size=1, shuffle=False)

# Load our checkpoint
# from model.segformer_trainer import SegFormer


model = ShadowVL(config)
pth_dict = torch.load(config["eval_model_path"])
# print(pth_dict.keys())
if "ckpt" in config["eval_model_path"]:
    pth_dict = pth_dict["model"]
model.load_state_dict(pth_dict, strict=False)
model.cuda().eval()
print(f'network loaded from {config["eval_model_path"]}!')


toImage = ToPILImage()


def eval_visha(config):
        
    total_process_time = 0
    total_frames = 0

    # Start eval
    # loader = DataLoader(vid_reader, batch_size=5, shuffle=False, num_workers=2)
    # vid_name = vid_reader.vid_name
    # vid_length = len(loader)
    video_ids = set()
    for data in progressbar(
        meta_loader, max_value=len(meta_dataset), redirect_stdout=True
    ):
        with torch.cuda.amp.autocast(enabled=config["amp"]):
            image = data["image"].cuda()  # b,5,3,h,w
            label_path = data["label_path"]
            descriptions = data["descriptions"]
            paths = []
            video_id = label_path[0][0].split('/')[-2]
            # print(label_path[0][0].split('/')[-2])
            if video_id not in video_ids:
                # if len(video_ids) > 0:
                #     model.clear_memory()
                video_ids.add(video_id)
            # print(label_path[0][0].split('/')[-2])
            for path in label_path:
                ls = path[0].split("/")
                paths.append((ls[-2], ls[-1]))
            # print(paths)

            shape = (data["h"][0], data["w"][0])  # original size
            b, t, c, h, w = image.shape
            # print(f'image:{image.shape}')
            # print(f'info:{info}')
            # print(f'shape:{shape}')

            """
            For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
            Seems to be very similar in testing as my previous timing method 
            with two cuda sync + time.time() in STCN though 
            """
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            logits,_ = model(image, text=descriptions, is_train=False,clear_memory=False)

            # Upsample to original size if needed
            # if need_resize:
            upscaled_mask = F.interpolate(
                logits, shape, mode="bilinear", align_corners=False
            )
            upscaled_mask = F.sigmoid(upscaled_mask)

            end.record()
            torch.cuda.synchronize()
            total_process_time += start.elapsed_time(end) / 1000
            total_frames += t

            # Probability mask -> index mask
            out_mask = torch.as_tensor(
                upscaled_mask > config["eval_threshold"], dtype=torch.float32
            )  # b,1,H,W

            # Save the mask
            for i in range(t):
                out_img = toImage(out_mask[i])
                this_out_path = os.path.join(config["eval_output_dir"], paths[i][0])
                os.makedirs(this_out_path, exist_ok=True)
                # out_img = Image.fromarray(out_mask,mode="L")
                # print(f'frame: {frame[i][:-4]}')
                out_img.save(os.path.join(this_out_path, paths[i][1]))
            # break
    # break

    print(f"Total processing time: {total_process_time}")
    print(f"Total processed frames: {total_frames}")
    print(f"FPS: {total_frames / total_process_time}")
    print(f"Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}")


eval_visha(config)

"""
    calculate scores
"""
print("---------now calculating scores---------")
Visha_scores(config,gpu=True)
