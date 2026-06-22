"""
Integrate numerical values for some iterations
Typically used for loss computation / logging to tensorboard
Call finalize and create a new Integrator when you want to display/log
"""

import torch


class Integrator:
    def __init__(self, logger, distributed=True, local_rank=0, world_size=1):
        self.values = {}  #
        self.counts = {}  # 记录每一个key对应元素的数量
        self.hooks = []  # List is used here to maintain insertion order

        self.logger = logger

        self.distributed = distributed
        self.local_rank = local_rank
        self.world_size = world_size

    def add_tensor(self, key, tensor):
        # 增加一个tensor
        if key not in self.values:
            # 这个大分支是新增加了一个key的情况
            self.counts[key] = 1
            if type(tensor) == float or type(tensor) == int:
                # 如果是一个数就直接添加
                self.values[key] = tensor
            else:
                # 如果是序列，就取平均成一个数再添加
                self.values[key] = tensor.mean().item()
        else:
            # 这个分支是向原有的key中添加数据的情况
            self.counts[key] += 1
            if type(tensor) == float or type(tensor) == int:
                # 直接累加
                self.values[key] += tensor
            else:
                # 取平均后再累加
                self.values[key] += tensor.mean().item()

    def add_dict(self, tensor_dict):
        # 批量添加key和value
        for k, v in tensor_dict.items():
            self.add_tensor(k, v)

    def add_hook(self, hook):
        # 自定义钩子函数，可以在finalize之前按照添加顺序进行数据处理
        """
        Adds a custom hook, i.e. compute new metrics using values in the dict
        The hook takes the dict as argument, and returns a (k, v) tuple
        e.g. for computing IoU
        """
        if type(hook) == list:
            self.hooks.extend(hook)
        else:
            self.hooks.append(hook)

    def reset_except_hooks(self):
        # 清零操作
        self.values = {}
        self.counts = {}

    # Average and output the metrics
    def finalize(self, prefix, it, f=None):
        # 该函数用于在添加好了所有值之后，进行统一的log操作

        for hook in self.hooks:
            # 执行hook
            k, v = hook(self.values)
            self.add_tensor(k, v)

        for k, v in self.values.items():
            if k[:4] == "hide":
                # 以hide为开头的key不会被log
                continue

            avg = v / self.counts[k]  # 平均

            if self.distributed:
                # Inplace operation
                avg = torch.tensor(avg).cuda()
                torch.distributed.reduce(avg, dst=0)  # 汇总各个rank上的值

                if self.local_rank == 0:
                    avg = (avg / self.world_size).cpu().item()  # 在rank间平均
                    self.logger.log_metrics(prefix, k, avg, it, f)  # log
            else:
                # 非DDP环境下直接log即可
                # Simple does it
                self.logger.log_metrics(prefix, k, avg, it, f)
