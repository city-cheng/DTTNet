import os
import torch
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Dataset
from .augmentations_video_decouple import (
    get_train_joint_transform,
    get_val_joint_transform,
)
import json


class ViSha_Dataset(Dataset):
    def __init__(self, mode: str, config: dict) -> None:
        self.config = config
        self.mode = mode
        self.is_training = mode == "train"
        print("Dataloader Mode:", "training" if self.is_training else "testing")

        # configs
        self.data_root = config["data_root"]
        self.image_folder = config["image_folder"]
        self.label_folder = config["label_folder"]
        self.image_ext = config["image_ext"]
        self.label_ext = config["label_ext"]

        # transform
        if self.is_training:
            self.joint_transform = get_train_joint_transform(
                scale=(config["scale"], config["scale"])
            )
            self.time_clips = config["time_clips"]
        else:
            self.joint_transform = get_val_joint_transform(
                scale=(config["scale"], config["scale"])
            )
            self.time_clips = config["time_clips"]

        # get all frames from video datasets
        self.frame_list, self.path_image, self.path_mask = (
            self.generate_images_from_video(is_training=self.is_training)
        )
        print("Total video clips are {}.".format(len(self.frame_list)))

        # 假设你的JSON文件名为 data.json
        llm = "/descriptions_qwen-vl-plus.json"
        # llm = "/descriptions.json"
        # llm = "/descriptions_qwen3-vl-plus.json"
        if self.is_training:
            file_path = self.data_root + "/train" + llm
            print("***********************Loading JSON file:", file_path)
        else:
            file_path = self.data_root + "/test" + llm
            print("***********************Loading JSON file:", file_path)

        # if self.is_training:
        #     file_path = "/share/datasets/ViSha_release/train/descriptions.json"
        # else:
        #     file_path = "/share/datasets/ViSha_release/test/descriptions.json"

        # 打开并读取JSON文件
        with open(file_path, "r", encoding="utf-8") as file:
            self.descriptions = json.load(file)

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        image_label_path_list = self.frame_list[index]

        clip_list = []
        label_list = []
        body_list = []
        detail_list = []
        w_list = []
        h_list = []
        image_path_list = []
        label_path_list = []
        for image_path, label_path, body_path, detail_path in image_label_path_list:
            if not self.is_training:
                image_path = self.path_image[image_path]
                label_path = self.path_mask[label_path]
                image = Image.open(image_path).convert("RGB")
                label = Image.open(label_path).convert("L")
                body = Image.open(body_path).convert("L")
                detail = Image.open(detail_path).convert("L")

            else:
                # image = self.path_image[image_path]
                # label = self.path_mask[label_path]
                image_path = self.path_image[image_path]
                label_path = self.path_mask[label_path]
                image = Image.open(image_path).convert("RGB")
                label = Image.open(label_path).convert("L")
                body = Image.open(body_path).convert("L")
                detail = Image.open(detail_path).convert("L").point(lambda x: 255 if x > 0 else 0)

            clip_list.append(image)
            label_list.append(label)
            w, h = image.size
            w_list.append(w)
            h_list.append(h)
            image_path_list.append(image_path)
            label_path_list.append(label_path)
            body_list.append(body)
            detail_list.append(detail)

        clip_list, label_list, body_list, detail_list = self.joint_transform(
            clip_list, label_list, body_list, detail_list
        )
        image_torch = torch.stack(clip_list)
        label_torch = torch.stack(label_list)
        body_torch = torch.stack(body_list)
        detail_torch = torch.stack(detail_list)

        video_name = image_path_list[0].split("/")[-2]
        # print(video_name)

        return {
            "image": image_torch,
            "label": label_torch,
            "body": body_torch,
            "detail": detail_torch,
            "image_path": image_path_list,
            "label_path": label_path_list,
            "w": w_list,
            "h": h_list,
            "descriptions": self.descriptions[video_name],
        }

    def generate_images_from_video(self, is_training=True):
        video_list = os.listdir(
            os.path.join(self.data_root, self.mode, self.image_folder)
        )
        video_frame_dict = {}
        path_frame_dict = {}
        path_mask_dict = {}

        for video in video_list:
            video_path = os.path.join(
                self.data_root, self.mode, self.image_folder, video
            )
            frame_list = [
                os.path.splitext(frame)[0]
                for frame in os.listdir(video_path)
                if frame.endswith(self.image_ext)
            ]
            frame_list = self.sort_images(frame_list)

            if self.is_training:
                # add more frames in revised order if less than 100 frames
                len_frame_list = len(frame_list)
                # if len_frame_list < 100:
                #     for _ in range(int(100/len_frame_list)+1):
                #         for reversed_frame in frame_list[-1:-(min(100-len_frame_list, len_frame_list)):-1]:
                #             frame_list.append(reversed_frame)
                #     if len(frame_list) >= 100:
                #         frame_list = frame_list[:100]

                # 改为保证帧数能被5除尽
                if len_frame_list % 5 != 0:
                    extra_frame = 5 - len_frame_list % 5
                    for reversed_frame in frame_list[-1 : -extra_frame - 1 : -1]:
                        frame_list.append(reversed_frame)

            video_frame_dict[video] = []
            for frame in frame_list:
                # frame_gt: (frame, gt)
                frame_path = os.path.join(
                    self.data_root,
                    self.mode,
                    self.image_folder,
                    video,
                    frame + self.image_ext,
                )
                gt_path = os.path.join(
                    self.data_root,
                    self.mode,
                    self.label_folder,
                    video,
                    frame + self.label_ext,
                )
                body_path = os.path.join(
                    self.data_root,
                    self.mode,
                    self.label_folder,
                    video,
                    "body-origin",
                    frame + self.label_ext,
                )
                detail_path = os.path.join(
                    self.data_root,
                    self.mode,
                    self.label_folder,
                    video,
                    "detail-origin",
                    frame + self.label_ext,
                )

                frame_gt = (frame_path, gt_path, body_path, detail_path)
                video_frame_dict[video].append(frame_gt)

                # if training, load data in init function in adcance. if testing, load data path for fast preprocessing.
                if is_training:
                    # path_frame_dict[frame_path] = Image.open(frame_path).convert('RGB')
                    # path_mask_dict[gt_path] = Image.open(gt_path).convert('L')
                    path_frame_dict[frame_path] = frame_path
                    path_mask_dict[gt_path] = gt_path
                else:
                    path_frame_dict[frame_path] = frame_path
                    path_mask_dict[gt_path] = gt_path

        # ensemble clips
        clip_list = []
        for video in video_list:
            frames_from_one_video = video_frame_dict[video]
            stride = 1 if self.is_training else self.time_clips
            for begin in range(
                0, len(frames_from_one_video) - self.time_clips + 1, stride
            ):
                frame_clips = frames_from_one_video[begin : begin + self.time_clips]
                clip_list.append(frame_clips)

            # last n image go backward for training, and last clip for test
            if self.is_training:
                for begin in range(
                    len(frames_from_one_video) - self.time_clips + 1,
                    len(frames_from_one_video),
                ):
                    frame_clips = frames_from_one_video[
                        begin : begin - self.time_clips : -1
                    ]
                    clip_list.append(frame_clips)
            else:
                last_frame_clips = frames_from_one_video[
                    len(frames_from_one_video) - self.time_clips :
                ]
                clip_list.append(last_frame_clips)

        return clip_list, path_frame_dict, path_mask_dict

    def sort_images(self, frame_list):
        frame_int_list = [int(frame) for frame in frame_list]
        # sort images to 001, 002, 003...
        sort_index = [
            i for i, v in sorted(enumerate(frame_int_list), key=lambda x: x[1])
        ]
        return [frame_list[i] for i in sort_index]

    def read_segmentation_mask(self, gt_path):
        gt_pil = Image.open(gt_path).convert("L")
        gt_np = np.array(gt_pil)

        # some gt are store in RGB, whose values are not [0, 255]
        if len(np.unique(gt_np)) != 2:
            gt_np[gt_np != 0] = 255

        return Image.fromarray(gt_np)
