import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
from collections import OrderedDict
from tqdm import tqdm, trange
import random
from scipy import integrate

torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda")

class params:
    def __init__(self):
        self.batch_size = 300
        self.epoches = 50000
        self.gamma = 0.99999
        self.num_hidden = 4
        self.width = 100
        self.num_hidden_R = 2
        self.width_R = 10
        self.n_bic_points = 100
        self.n_inner_points = int(np.sqrt(10000))
        self.n_points_test = 200
        self.Adam_proportion = 0.99
        self.coe_structure = 1.0
        self.tol = 1.0
        self.x_l = -15.0
        self.x_r = 15.0
        self.t_0 = 0.0
        self.t_T = math.pi / 2.0

class NLS:
    def __init__(self):
        self.C_1 = np.tanh(15) + np.tanh(15)
        self.C_2 = 4 * np.tanh(15)

    @staticmethod
    def IC(x):
        return 1.0 / torch.cosh(x) * torch.cos(2 * x), -1.0 / torch.cosh(x) * torch.sin(2 * x)

    @staticmethod
    def exact_solution(X):
        x = X[:,0]
        t = X[:,1]
        return 1.0 / torch.cosh(x + 2*t) * torch.cos(2*x + 3*t/2), -1.0 / torch.cosh(x + 2*t) * torch.sin(2*x + 3*t/2)
