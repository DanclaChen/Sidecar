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

        # hyperparameters
        # self.batch_size = 32
        self.batch_size = 2046
        self.epoches = 10000
        self.h_train = 0.02
        self.tau_train = 0.01
        self.gamma = 0.99999

        self.coe_without_R = 0.1

        self.width = 20
        self.num_hidden = 2

        self.width_R = 4
        self.num_hidden_R = 1

        self.Adam_proportion = 0.95


class Burgers:
    def __init__(self):
        # parameters of the Burger's equation
        self.nu = 0.1

    # the initial condition
    def IC(self, x):
        # return -torch.sin(math.pi * x)
        return 2 * math.pi * self.nu * (torch.sin(math.pi * x)) / (2 + torch.cos(math.pi * x))

    def exact_solution(self, X):
        x = X[:,0]
        t = X[:,1]
        
        return 2 * math.pi * self.nu * (torch.exp(- math.pi**2 * self.nu * t) * torch.sin(math.pi * x)) / (2 + torch.exp(- math.pi**2 * self.nu * t) * torch.cos(math.pi * x))