"""
trainer.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from utils.losses import SASLossComputer
from utils.log_integrator import Integrator

from utils.transforms import inv_im_normalization

from model.shadowvl import ShadowVL
# from model.shadowvl_v2 import ShadowVL2


class SASTrainer:
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1):
        self.config = config
        self.local_rank = local_rank

        # TODO  DDP模式
        self.model = nn.parallel.DistributedDataParallel(
            ShadowVL(config).cuda(),
            # ShadowVL2(config).cuda(),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=config["find_unused_parameters"],
        )

        # Set up logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            # 计时
            self.last_time = time.time()
            # 参数量
            self.logger.log_string(
                "model_size",
                str(sum([param.nelement() for param in self.model.parameters()])),
            )
            for name, param in self.model.module.named_parameters():
                if param.requires_grad:
                    self.logger.log_string(
                        "trainable_params",
                        str(name) + ":" + str(param.nelement()),
                    )
        self.train_integrator = Integrator(
            self.logger, distributed=True, local_rank=local_rank, world_size=world_size
        )
        self.loss_computer = SASLossComputer(config)

        self.train()

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config["scratch_learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, config["steps"], config["gamma"]
        )
        if config["amp"]:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config["log_text_interval"]
        self.log_image_interval = config["log_image_interval"]
        self.save_network_interval = config["save_network_interval"]
        self.save_checkpoint_interval = config["save_checkpoint_interval"]
        self.check_grad_interval = config["check_grad_interval"]
        if config["debug"]:
            self.log_text_interval = 1
            self.log_image_interval = 1
            self.save_network_interval = 5
            self.save_checkpoint_interval = 5
            self.check_grad_interval = 1

    def do_pass(self, data, it=0):

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        # 准备一次迭代的数据
        images = data["image"]  # b,t,3,h,w
        gt = data["label"]  # b,t,1,h,w
        x_bound_gt = data["x_bound"]  # b,t,1,h,w
        # print(x_bound_gt.shape,x_bound_gt.dtype)
        descriptions = data["descriptions"]
        # descriptions = [descriptions['shadow']]

        # AMP，半精度训练
        with torch.cuda.amp.autocast(enabled=self.config["amp"]):

            # (logits, bounds), preds = self.model(images, gt, descriptions)
            logits, bounds = self.model(images, gt, descriptions)

            logits = torch.nn.functional.interpolate(
                logits, size=self.config["scale"], mode="bilinear", align_corners=False
            )  # b*t,1,h,w
            # for i in range(len(bounds)):
            bounds = torch.nn.functional.interpolate(
                bounds, size=self.config["scale"], mode="bilinear", align_corners=False
            )  # b*t,1,h,w
            gt = gt.flatten(0, 1)
            images = images.flatten(0, 1)
            x_bound_gt = x_bound_gt.flatten(0, 1)
            if self._do_log or self._is_train:
                # 如果需要log
                # 计算损失，返回一个字典
                losses = self.loss_computer.compute(
                    logits,
                    gt,
                    coarse_mask=bounds,
                    x_bound_gt=x_bound_gt,
                    preds=None,
                    it=it,
                )
                # losses['contrast_loss'] = contrast_loss
                # losses["total_loss"] += 0.1 * losses["contrast_loss"]

                # Logging pictures
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.log_image_interval == 0 and it != 0:
                            if self.logger is not None:
                                # 记录图片
                                size = (384, 384)
                                self.logger.log_cv2(
                                    "train/images",
                                    inv_im_normalization(images),
                                    it,
                                    size=size,
                                )
                                self.logger.log_cv2(
                                    "train/preds",
                                    torch.tensor(logits > 0, dtype=torch.int8),
                                    it,
                                    size=size,
                                )
                                self.logger.log_cv2("train/labels", gt, it, size=size)
                                # if coarse_mask is not None:
                                #     self.logger.log_cv2(
                                #         "train/coarse_mask",
                                #         torch.tensor(coarse_mask > 0, dtype=torch.int8),
                                #         it,
                                #         size=size,
                                #     )
                                self.logger.log_cv2(
                                    "train/x_bound",
                                    torch.tensor(bounds > 0, dtype=torch.int8),
                                    it,
                                    size=size,
                                )
                                self.logger.log_cv2(
                                    "train/x_bound_gt",
                                    x_bound_gt,
                                    it,
                                    size=size,
                                )

            # 记录训练信息
            if self._is_train:
                if (it) % self.log_text_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar(
                            "train/lr", self.scheduler.get_last_lr()[0], it
                        )
                        self.logger.log_metrics(
                            "train",
                            "time",
                            (time.time() - self.last_time) / self.log_text_interval,
                            it,
                        )
                    self.last_time = time.time()
                    self.train_integrator.finalize("train", it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_network(it)

                if it % self.save_checkpoint_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_checkpoint(it)

        # Backward pass  反向传播
        self.optimizer.zero_grad(set_to_none=True)
        if self.config["amp"]:
            self.scaler.scale(losses["total_loss"]).backward(retain_graph=True)
            # self.scaler.scale(losses["total_loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses["total_loss"].backward(retain_graph=True)
            self.optimizer.step()

        # 检查梯度
        if it % self.check_grad_interval == 0 and it != 0:
            for name, param in self.model.module.named_parameters():
                if (
                    param.grad == None
                    and "feat_net" not in name
                    and param.requires_grad
                ):
                    print("----------检测到有参数梯度消失----------")
                    print(name)
                    if self.config["debug"] == False:
                        break
                if (
                    param.grad != None
                    and torch.isinf(param.grad).any()
                    and param.requires_grad
                ):
                    print("----------检测到有参数梯度爆炸----------")
                    print(name)
                    if self.config["debug"] == False:
                        break

        self.scheduler.step()

    def save_network(self, it):
        # 保存网络
        if self.save_path is None:
            print("Saving has been disabled.")
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f"{self.save_path}_{it}.pth"
        torch.save(self.model.module.state_dict(), model_path)
        print(f"Network saved to {model_path}.")

    def save_checkpoint(self, it):
        # 保存记录点
        if self.save_path is None:
            print("Saving has been disabled.")
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f"{self.save_path}_checkpoint_{it}.pth"
        # TODO 记录点包含以下信息
        checkpoint = {
            "it": it,
            "network": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}.")

    def load_checkpoint(self, path):
        # 读取模型
        # This method loads everything and should be used to resume training
        map_location = "cuda:%d" % self.local_rank
        # 把第0rank的模型读到当前rank上
        checkpoint = torch.load(path, map_location={"cuda:0": map_location})
        # 恢复训练信息
        it = checkpoint["it"]
        network = checkpoint["network"]
        optimizer = checkpoint["optimizer"]
        scheduler = checkpoint["scheduler"]

        map_location = "cuda:%d" % self.local_rank
        self.model.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print("Network weights, optimizer states, and scheduler states loaded.")

        return it

    def load_network_in_memory(self, src_dict):
        self.model.module.load_state_dict(src_dict, strict=False)

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = "cuda:%d" % self.local_rank
        src_dict = torch.load(path, map_location={"cuda:7": map_location})
        if "ckpt" in path:
            src_dict = src_dict["model"]

        self.load_network_in_memory(src_dict)
        print(f"Network weight loaded from {path}")

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.model.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.model.eval()
        return self
