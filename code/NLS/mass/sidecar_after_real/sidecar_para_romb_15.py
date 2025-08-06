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

from tensorboardX import SummaryWriter

from scipy import integrate

from parameters_15 import NLS

torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda")

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class PDE_NN(nn.Module):
    def __init__(
            self,
            width,
            num_hidden,
            input_size = 2,
            hidden_size = None,
            output_size = 2,
            act = nn.Tanh,
        ):
        super(PDE_NN, self).__init__()
        if width is not None:
            hidden_size = [width] * num_hidden
        elif isinstance(hidden_size, list) and len(hidden_size) == num_hidden:
            pass
        else:
            raise ValueError("Invalid hidden size specification")
        layers = [('input', nn.Linear(input_size, hidden_size[0]))]
        layers.append(('input_activation', act()))
        for i in range(1, num_hidden):
            layers.append(
                ('hidden_%d' % i, nn.Linear(hidden_size[i-1], hidden_size[i]))
            )
            layers.append(('activation_%d' % i, act()))
        layers.append(('output', nn.Linear(hidden_size[-1], output_size)))
        layerDict = OrderedDict(layers)
        self.layers = nn.Sequential(layerDict)
    def forward(self, X):
        out = self.layers(X)
        return out

class structure_NN(nn.Module):
    def __init__(
            self,
            width_R,
            num_hidden_R,
            input_size = 1,
            hidden_size = None,
            output_size = 1,
            act = nn.Tanh,
        ):
        super(structure_NN, self).__init__()
        if width_R is not None:
            hidden_size = [width_R] * num_hidden_R
        elif isinstance(hidden_size, list) and len(hidden_size) == num_hidden_R:
            pass
        else:
            raise ValueError("Invalid hidden size specification")
        layers = [('input', nn.Linear(input_size, hidden_size[0]))]
        layers.append(('input_activation', act()))
        for i in range(1, num_hidden_R):
            layers.append(
                ('hidden_%d' % i, nn.Linear(hidden_size[i-1], hidden_size[i]))
            )
            layers.append(('activation_%d' % i, act()))
        layers.append(('output', nn.Linear(hidden_size[-1], output_size)))
        layerDict = OrderedDict(layers)
        self.layers = nn.Sequential(layerDict)
    def forward(self, X):
        out = self.layers(X)
        return out

class NN(nn.Module):
    def __init__(self, width, num_hidden, width_R, num_hidden_R):
        super(NN, self).__init__()
        self.PDE = PDE_NN(width, num_hidden)
        self.structure = structure_NN(width_R, num_hidden_R)
    def forward(self, X):
        v = self.PDE(X)
        R = self.structure(X[:,1].reshape(-1, 1))
        return v, R

class bic_Dataset(Dataset):
    def __init__(self, n_points, x_l, x_r, t_0, t_T):
        self.n_points = n_points
        x_initial = torch.linspace(x_l, x_r, n_points * 4 + 1).unsqueeze(1)
        t_initial = torch.zeros(n_points * 4 + 1, 1)
        t_boundary = torch.linspace(t_0, t_T, n_points).unsqueeze(1)
        x_boundary = torch.full((n_points, 1), x_l)
        self.x = torch.cat([x_initial, x_boundary], dim=0)
        self.t = torch.cat([t_initial, t_boundary], dim=0)
        self.labels = torch.cat([torch.ones(n_points * 4 + 1), torch.full((n_points,), 2)], dim=0).long()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.labels[idx]

class structured_PINNs():
    def __init__(self, params, j, device):
        self.j = j
        self.learning_rate = params.learning_rate
        self.batch_size = params.batch_size
        self.epoches = params.epoches
        self.Adam_proportion = params.Adam_proportion
        self.gamma = params.gamma
        self.n_bic_points = params.n_bic_points
        self.n_inner_points = params.n_inner_points
        self.n_points_test = params.n_points_test
        self.coe_structure = params.coe_structure
        self.tol = params.tol
        self.NLS = NLS()
        self.device = device
        self.width = params.width
        self.num_hidden = params.num_hidden
        self.width_R = params.width_R
        self.num_hidden_R = params.num_hidden_R
        self.model = NN(self.width, self.num_hidden, self.width_R, self.num_hidden_R).to(self.device)
        self.model.load_state_dict(torch.load(f"/home/22040517r/sidecar/Sidecar/codes/NLS/parallel/moving/PINNs_withR/trained_model/NLS-romb-bs{self.batch_size}-epoches100000-width{self.width}-depth{self.num_hidden}-widthR{self.width_R}-depthR{self.num_hidden_R}-n_bic{self.n_bic_points}-n_inner{self.n_inner_points}-gamma0.99999-{self.j}.pth", map_location= f"cuda:{self.device}"))
        self.x_l = params.x_l
        self.x_r = params.x_r
        self.t_0 = params.t_0
        self.t_T = params.t_T
        self.x_train = torch.linspace(self.x_l, self.x_r, self.n_inner_points * 4 + 1)
        self.t_train = torch.linspace(self.t_0, self.t_T, self.n_inner_points)
        self.X_train = torch.stack(torch.meshgrid(self.x_train, self.t_train, indexing='ij')).reshape(2, -1).T.to(self.device)
        self.X_train.requires_grad = True
        bic_dataset = bic_Dataset(self.n_bic_points, self.x_l, self.x_r, self.t_0, self.t_T)
        self.bic_loader = DataLoader(bic_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        self.bic_loader_LBFGS = DataLoader(bic_dataset, batch_size = len(bic_dataset), generator=torch.Generator(device='cuda'))
        self.criterion = torch.nn.MSELoss()
        self.optimizer_adam = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_adam, self.gamma)
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )
    def complex_multiply(self, a, b):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            a_re, a_im = a[:, 0], a[:, 1]
            b_re, b_im = b[:, 0], b[:, 1]
            real_part = a_re * b_re - a_im * b_im
            imag_part = a_re * b_im + a_im * b_re
            return np.stack((real_part, imag_part), axis=1)
        elif torch.is_tensor(a) and torch.is_tensor(b):
            a_re, a_im = torch.unbind(a, dim=1)
            b_re, b_im = torch.unbind(b, dim=1)
            real_part = a_re * b_re - a_im * b_im
            imag_part = a_re * b_im + a_im * b_re
            return torch.stack((real_part, imag_part), dim=1)
        else:
            raise TypeError("Inputs must be both NumPy arrays or both PyTorch tensors")
    def gradient(self,func,var,order = 1):
        if order == 1:
            return torch.autograd.grad(inputs = var,
                                       outputs = func,
                                       grad_outputs=torch.ones_like(func),
                                       create_graph=True,
                                       only_inputs=True)[0]
        else:
            out = self.gradient(func,var)
            return self.gradient(out,var,order - 1)
    def loss_fn(self, X, x_inner, t_inner, x, t, labels, epoch):
        if len(x[labels == 2]) == 0:
            loss_bc = torch.tensor(0.0, requires_grad=True).to(self.device)
        else:
            x_bc1 = torch.full_like(x[labels == 2], self.x_l, requires_grad=True)
            x_bc2 = torch.full_like(x[labels == 2], self.x_r, requires_grad=True)
            t_bc = t[labels == 2]
            v_pred_bc1, R_bc1 = self.model(torch.cat([x_bc1, t_bc], dim=1))
            v_pred_bc2, R_bc2 = self.model(torch.cat([x_bc2, t_bc], dim=1))
            u_pred_bc1_re, u_pred_bc1_im = torch.unbind(v_pred_bc1 * R_bc1, dim=1)
            u_pred_bc2_re, u_pred_bc2_im = torch.unbind(v_pred_bc2 * R_bc2, dim=1)
            du_dx_bc1_re = self.gradient(u_pred_bc1_re, x_bc1)[:, 0]
            du_dx_bc1_im = self.gradient(u_pred_bc1_im, x_bc1)[:, 0]
            du_dx_bc2_re = self.gradient(u_pred_bc2_re, x_bc2)[:, 0]
            du_dx_bc2_im = self.gradient(u_pred_bc2_im, x_bc2)[:, 0]
            loss_bc = self.criterion(u_pred_bc1_re, u_pred_bc2_re) + self.criterion(u_pred_bc1_im, u_pred_bc2_im) + self.criterion(du_dx_bc1_re, du_dx_bc2_re) + self.criterion(du_dx_bc1_im, du_dx_bc2_im)
        if len(x[labels == 1]) == 0:
            loss_ic = torch.tensor(0.0, requires_grad=True).to(self.device)
        else:
            X_ic = torch.cat([x[labels == 1], t[labels == 1]], dim=1)
            u_pred_ic, R_ic = self.model(X_ic)
            u_pred_ic_re, u_pred_ic_im = torch.unbind(u_pred_ic * R_ic, dim=1)
            u_ic_re, u_ic_im = NLS.IC(X_ic[:, 0])
            loss_ic = self.criterion(u_pred_ic_re, u_ic_re) + self.criterion(u_pred_ic_im, u_ic_im)
        loss_data = loss_bc + loss_ic
        v_pred, R = self.model(X)
        u_pred_re, u_pred_im = torch.unbind(v_pred * R, dim=1)
        du_dX_re = self.gradient(u_pred_re, X)
        du_dX_im = self.gradient(u_pred_im, X)
        du_dt_re = du_dX_re[:, 1]
        du_dt_im = du_dX_im[:, 1]
        du_dxx_re = self.gradient(du_dX_re[:, 0], X)[:, 0]
        du_dxx_im = self.gradient(du_dX_im[:, 0], X)[:, 0]
        u_norm_square = u_pred_re**2 + u_pred_im**2
        loss_re = self.criterion(-du_dt_im + 0.5 * du_dxx_re + u_norm_square * u_pred_re, torch.zeros(u_pred_re.size()).to(self.device))
        loss_im = self.criterion(du_dt_re + 0.5 * du_dxx_im + u_norm_square * u_pred_im, torch.zeros(u_pred_re.size()).to(self.device))
        loss_pde = loss_re + loss_im
        v_pred_re, v_pred_im = torch.unbind(v_pred, dim=1)
        integral_temp = v_pred_re ** 2 + v_pred_im ** 2
        integral_temp_matrix = integral_temp.reshape(len(x_inner), len(t_inner)).detach().cpu().numpy()
        h = x_inner[1] - x_inner[0]
        integral = torch.from_numpy(integrate.romb(integral_temp_matrix.T, dx = h.cpu().numpy())).to(self.device)
        mass_pred = integral * (R[:len(t_inner)]**2).squeeze()
        loss_structure_vector = (mass_pred - torch.tensor(self.NLS.C_1).repeat(len(t_inner)).to(self.device))**2
        M_sum = torch.triu(torch.ones(len(t_inner), len(t_inner)), diagonal=1).T
        with torch.no_grad():
            W = torch.exp(-self.tol * torch.matmul(M_sum, loss_structure_vector))
        loss_structure_vector_causal = W * loss_structure_vector
        loss_structure = torch.mean(loss_structure_vector_causal)
        u_square = u_pred_re ** 2 + u_pred_im ** 2
        u_square_matrix = u_square.reshape(len(x_inner), len(t_inner)).detach().cpu().numpy()
        temp = integrate.romb(u_square_matrix.T, dx = h.cpu().numpy())
        loss_energy_vector = np.abs(temp - self.NLS.C_1)
        loss_energy = np.max(loss_energy_vector)
        u_exact_re, u_exact_im = NLS.exact_solution(X)
        loss_exact = self.criterion(u_pred_re, u_exact_re) + self.criterion(u_pred_im, u_exact_im)
        loss = loss_pde + loss_data + self.coe_structure * loss_structure
        return loss, loss_pde, loss_data, loss_energy, loss_exact, loss_structure
    def closure(self):
        self.optimizer_LBFGS.zero_grad()
        loss_vector = self.loss_fn(self.X_train, self.x_train, self.t_train, self.x_LBFGS, self.t_LBFGS, self.labels_LBFGS, self.current_epoch)
        loss_vector[0].backward()
        return loss_vector[0]
    def train(self):
        writer = SummaryWriter(comment=f"NLS-withR-bs{self.batch_size}-epoches{self.epoches}-width{self.width}-n_bic{self.n_bic_points}-n_inner{self.n_inner_points}-gamma{self.gamma}-{self.j}")
        self.model.train()
        for epoch in range(self.epoches):
            if epoch < (self.Adam_proportion * self.epoches):
                for x, t, labels in self.bic_loader:
                    x, t, labels = x.to(self.device), t.to(self.device), labels.to(self.device)
                    self.optimizer_adam.zero_grad()
                    loss_vector = self.loss_fn(self.X_train, self.x_train, self.t_train, x, t, labels, epoch)
                    loss_vector[0].backward()
                    self.optimizer_adam.step()
                    self.scheduler.step()
            else:
                for x, t, labels in self.bic_loader_LBFGS:
                    self.x_LBFGS, self.t_LBFGS, self.labels_LBFGS = x.to(self.device), t.to(self.device), labels.to(self.device)
                    self.current_epoch = epoch
                    self.optimizer_LBFGS.step(self.closure)
                    loss_vector = self.loss_fn(self.X_train, self.x_train, self.t_train, self.x_LBFGS, self.t_LBFGS, self.labels_LBFGS, epoch)
            writer.add_scalar('Train/Total loss', loss_vector[0], epoch)
            writer.add_scalar('Train/PDE loss', loss_vector[1], epoch)
            writer.add_scalar('Train/data loss', loss_vector[2], epoch)
            writer.add_scalar('Train/energy loss', loss_vector[3], epoch)
            writer.add_scalar('Train/exact loss', loss_vector[4], epoch)
            writer.add_scalar('Train/structure loss', loss_vector[5], epoch)
            writer.add_scalar('Train/Learning Rate',self.optimizer_adam.state_dict()['param_groups'][0]['lr'],epoch)
        if not os.path.exists('./trained_model'):
            os.makedirs('./trained_model')
        torch.save(self.model.state_dict(), f"./trained_model/NLS-after-15-romb-lr{self.learning_rate}-bs{self.batch_size}-epoches{self.epoches}-width{self.width}-depth{self.num_hidden}-widthR{self.width_R}-depthR{self.num_hidden_R}-n_bic{self.n_bic_points}-n_inner{self.n_inner_points}-coeR{self.coe_structure}-gamma{self.gamma}-{self.j}.pth")
        writer.close()
    def evaluation(self):
        x_test = torch.linspace(self.x_l, self.x_r, self.n_points_test * 4 + 1)
        t_test = torch.linspace(self.t_0, self.t_T, self.n_points_test)
        X_test = torch.stack(torch.meshgrid(x_test, t_test, indexing='ij')).reshape(2, -1).T
        X_test = X_test.to(self.device)
        X_test.requires_grad = True
        x_ic_test = x_test.unsqueeze(1)
        t_ic_test = torch.zeros(self.n_points_test * 4 + 1, 1)
        t_bc_test = t_test.unsqueeze(1)
        x_bc_test = torch.full((self.n_points_test, 1), self.x_l)
        x = torch.cat([x_ic_test, x_bc_test], dim=0).to(self.device)
        t = torch.cat([t_ic_test, t_bc_test], dim=0).to(self.device)
        labels = torch.cat([torch.ones(self.n_points_test * 4 + 1), torch.full((self.n_points_test,), 2)], dim=0).long().to(self.device)
        self.model.eval()
        loss_vector = self.loss_fn(X_test, x_test, t_test, x, t, labels, self.epoches)
        self.model.zero_grad()
        return torch.tensor(loss_vector)

def parallel_train(j, params, results_queue, device):
    torch.set_default_dtype(torch.float64)
    torch.cuda.set_device(device)
    setup_seed(int(2023 + (1000 * j)))
    net = structured_PINNs(params, j, device)
    loss_before = net.evaluation().to('cpu')
    net.train()
    loss_after = net.evaluation().to('cpu')
    result = torch.cat([loss_after, loss_before])
    results_queue.put((j, result))
