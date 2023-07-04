# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import torch
from numpy import random
import numpy as np

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

seed_torch()
