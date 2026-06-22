import datetime
from os import path
import math
import os
from config.load import load_config


# 读取配置
# config_path = "./config/cvsd_config.yaml"
config_path = "./config/visha_config.yaml"
config = load_config(config_path)

# TODO 设置gpu优先级
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as distributed

# from model.trainer import SASTrainer
from model.trainer_decouple import SASTrainer

from utils.logger import TensorboardLogger

# from data.visha_dataset_video import ViSha_Dataset
from data.visha_dataset_decouple import ViSha_Dataset

#################################################################################################

"""
Initial setup
"""


# Init distributed environment
distributed.init_process_group(backend="nccl")
print(f"CUDA Device count: {torch.cuda.device_count()}")

# Parse command line arguments

if config["debug"]:
    config["exp_id"] = config["exp_id"] + "_debug"


# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

git_info = config["exp_id"]

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(f"I am rank {local_rank} in this world of size {world_size}!")

# Set seed to ensure the same initialization
# 设置各种随机种子
if config["seed"] != None:
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])


config["num_gpus"] = world_size
if (
    config["batch_size"] // config["num_gpus"] * config["num_gpus"]
    != config["batch_size"]
):
    raise ValueError("Batch size must be divisible by the number of GPUs.")
config["batch_size"] //= config["num_gpus"]
config["num_workers"] //= config["num_gpus"]
print(f'We are assuming {config["num_gpus"]} GPUs.')
#################################################################################################
"""
Model related
"""
if local_rank == 0:
    # Logging
    if config["exp_id"].lower() != "null":
        print("I will take the role of logging!")
        long_id = "%s_%s" % (
            datetime.datetime.now().strftime("%b%d_%H.%M.%S"),
            config["exp_id"],
        )
    else:
        long_id = None
    logger = TensorboardLogger(config["exp_id"], long_id, git_info)
    logger.log_string(
        "hyperparams", str(config)
    )  # 这里把配置项转成字符串，作为超参数log到tensorboard中

    # Construct the rank 0 model
    # rank 0的模型需要指定保存路径和logger
    model = SASTrainer(
        config,
        logger=logger,
        save_path=(
            path.join(".", "saves", long_id, long_id) if long_id is not None else None
        ),
        local_rank=local_rank,
        world_size=world_size,
    ).train()
else:
    # Construct model for other ranks
    # 其他rank上的模型不需要log和保存
    model = SASTrainer(config, local_rank=local_rank, world_size=world_size).train()

total_iter = 0
#################################################################################################
"""
Dataloader related
定义生成dataloader的函数
"""


# To re-seed the randomness everytime we start a worker
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 100
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def construct_loader(dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, rank=local_rank, shuffle=True
    )
    train_loader = DataLoader(
        dataset,
        config["batch_size"],
        sampler=train_sampler,
        num_workers=config["num_workers"],
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    return train_sampler, train_loader


def visha_loader(mode="train"):
    visha_dataset = ViSha_Dataset(mode, config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        visha_dataset, rank=local_rank, shuffle=True
    )
    if mode == "train":
        dataloader = DataLoader(
            visha_dataset,
            batch_size=config["batch_size"],
            sampler=train_sampler,
            num_workers=config["num_workers"],
            worker_init_fn=worker_init_fn,
        )
    else:
        dataloader = DataLoader(
            visha_dataset,
            batch_size=config["batch_size"],
            sampler=train_sampler,
            num_workers=config["num_workers"],
            worker_init_fn=worker_init_fn,
        )
    return train_sampler, dataloader


#################################################################################################
"""
Dataset related
"""
# 定义sampler，loader
train_sampler, train_loader = visha_loader()
#################################################################################################


"""
Determine max epoch
根据设置的最大iter计算最大epoch
"""
# debug模式
if config["debug"]:
    config["iterations"] = 5
total_epoch = math.ceil(config["iterations"] / len(train_loader))
current_epoch = total_iter // len(train_loader)
print(f"We approximately use {total_epoch} epochs.")
#################################################################################################


"""
Starts training
"""
finetuning = False
# Need this to select random bases in different workers
np.random.seed(np.random.randint(2**30 - 1) + local_rank * 100)
try:
    while total_iter < config["iterations"]:
        # Crucial for randomness!
        train_sampler.set_epoch(current_epoch)
        current_epoch += 1
        print(f"Current epoch: {current_epoch}")

        # Train loop
        model.train()
        for data in train_loader:
            # 一个batch中的操作
            # do_pass封装了前向，反向，loss，update等操作
            model.do_pass(data, total_iter)
            total_iter += 1

            if total_iter >= config["iterations"]:
                break
finally:
    if model.logger is not None and total_iter > 5000:
        model.save_network(total_iter)
        model.save_checkpoint(total_iter)


distributed.destroy_process_group()
