import torch
import torch.nn as nn
import random
import numpy as np
import os

from train import train_fn
from test import test_fn
from utils.build_loaders import build_loaders
from utils.build_model import build_model
from configs.settings import get_settings
from models.hungary import HungarianMatcher

def set_random_seed(seed):
    random.seed(seed)
    # ZFC 统一各Python Interpreter之间的Hash计算方法，保证它们在计算同一个对象时hash值相同，参考：https://zhuanlan.zhihu.com/p/456306448
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ZFC 以下二者相配合 --> 保证卷积计算的一致性
    torch.backends.cudnn.deterministic = True  # 固定的卷积算法因其实现不同 --> 也可能不受控制(结果可能有细微差别)，deterministic = True 保证使用确定性的卷积算法
    torch.backends.cudnn.benchmark = False  # ZFC benchmark = False -- 保证使用固定的卷积算法 --> 禁用卷积算法的选择机制


if __name__ == '__main__':
    cfgs = get_settings()
    set_random_seed(seed=cfgs.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, valid_loader, test_loader = build_loaders(cfgs)

    hungary_matcher = HungarianMatcher()
    model = build_model(cfgs)
    model = model.float()
    model.to(device)

    model = train_fn(cfgs, cfgs.model_name, model, hungary_matcher, train_loader, valid_loader, device)
    model.load_state_dict(torch.load(cfgs.train.save_checkpoints_path))
    model.eval()
    test_fn(cfgs, model, test_loader, device)

