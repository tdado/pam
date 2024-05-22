import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from scipy import stats
from scipy.stats import t
from numpy.linalg import svd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

# StyleGAN-XL
import sys
sys.path.append('../stylegan_xl')
from metrics import metric_utils
import torch.nn.functional as F
import dnnlib
import legacy
import torch