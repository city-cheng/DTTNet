import torch
import numpy as np
import random


def reseed(seed):
    """
    重新指定随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
