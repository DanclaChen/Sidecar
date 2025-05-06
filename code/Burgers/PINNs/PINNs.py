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


from parameters import Burgers

torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda")


def setup_seed(seed):
    
    random.seed(seed)   
    os.environ['PYTHONHASHSEED'] = str(seed)   
    np.random.seed(seed)   
    torch.manual_seed(seed)   
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True

    
# the whole network is divide into two parts    

# the subnetwork followed the standard PINNs to get the solution v of the PDE 
class PDE_NN(nn.Module):
    def __init__(
            self,
            width,
            num_hidden,
            input_size = 2,
            # hidden_size = [20, 20, 20, 20, 20],
            hidden_size = None,
            output_size = 1,
            act = nn.Tanh,
        ):
        super(PDE_NN, self).__init__()

        # if width is provided
        if width is not None:
            hidden_size = [width] * num_hidden
        # if hidden_size is provided
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
    
    
# the whole network with a output as u = R * v
# to add the regularization term, we have to give the two output separately
class NN(nn.Module):
    def __init__(self, width, num_hidden):
        super(NN, self).__init__()
        self.PDE = PDE_NN(width, num_hidden)

    def forward(self, X):
        u = self.PDE(X)
        return u





class structured_PINNs():
    def __init__(self, params, j):

        # multiprocessing index
        self.j = j
        
        # hyperparameters
        self.learning_rate = 1e-3
        self.batch_size = params.batch_size
        self.epoches = params.epoches
        self.Adam_proportion = params.Adam_proportion
        self.gamma = params.gamma
        self.coe_without_R = params.coe_without_R

        self.h_train = params.h_train
        self.tau_train = params.tau_train

        # parameters of the Burger's equation
        self.Burgers = Burgers()
        self.nu = self.Burgers.nu
        
        # initialization
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.width = params.width
        self.num_hidden = params.num_hidden
        self.model = NN(self.width, self.num_hidden).to(self.device)
        
        # time and space variables, where the external point is for bic
        self.x_train = torch.arange(-1, 1 + params.h_train, params.h_train)
        self.t_train = torch.arange(0, 1 + params.tau_train, params.tau_train)
        
        # the inner training points of spatial–temporal space
        # equidistance sample
        self.X_train = torch.stack(torch.meshgrid(self.x_train, self.t_train, indexing='ij')).reshape(2, -1).T.to(self.device)
        self.X_train.requires_grad = True

        # print(f"X: ", self.X[:,1])
        
        
        # lozation of training data
        bc1 = torch.stack(torch.meshgrid(self.x_train[0], self.t_train, indexing='ij')).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(self.x_train[-1], self.t_train, indexing='ij')).reshape(2, -1).T
        ic = torch.stack(torch.meshgrid(self.x_train, self.t_train[0], indexing='ij')).reshape(2, -1).T
        X_train_bic = torch.cat([bc1, bc2, ic]).to(self.device)
        
        
        # the corresponding boundary and initial condition
        u_bc1 = torch.zeros(len(bc1))
        u_bc2 = torch.zeros(len(bc2))
        u_ic = self.Burgers.IC(ic[:, 0])
        u_bic = torch.cat([u_bc1, u_bc2, u_ic]).unsqueeze(1).to(self.device)
        
        
        # training set
        train_set = torch.utils.data.TensorDataset(X_train_bic, u_bic)
        self.train_dataloader = DataLoader(train_set, batch_size = self.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        self.train_dataloader_LBFGS = DataLoader(train_set, batch_size = len(train_set), generator=torch.Generator(device='cuda'))

        
        
        # loss function measured by L^2 norm
        self.criterion = torch.nn.MSELoss()
        
        # optimizers
        self.optimizer_adam = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_adam, self.gamma)
        
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(), 
            lr=1.0, 
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-7, 
            # tolerance_change=1.0 * np.finfo(float).eps,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",   # better numerical stability
        )

    
    # the loss function, which contains four parts: PDE, boundary and initial condition, structure factor equation, regularization
    def loss_fn(self, u_pred_bic, u_bic, u_pred, X, x, t, epoch):
        
        
####### 1. the bc and ic error
        loss_data = self.criterion(u_pred_bic, u_bic)

        
####### 2. PDE error       
        # the derivative 
        du_dX = torch.autograd.grad(
                inputs = X, 
                outputs = u_pred, 
                grad_outputs = torch.ones_like(u_pred), 
                retain_graph = True, 
                create_graph = True
            )[0]

        du_dt = du_dX[:, 1]
        du_dx = du_dX[:, 0]
        du_dxx = torch.autograd.grad(
                inputs = X, 
                outputs = du_dx, 
                grad_outputs = torch.ones_like(du_dx), 
                retain_graph = True, 
                create_graph = True
            )[0][:, 0]

        # the L^2 norm of the PDE residual
        loss_pde = self.criterion(du_dt + u_pred.squeeze() * du_dx, self.nu * du_dxx)
        
        u_exact_matrix = self.Burgers.exact_solution(X).reshape(len(x), len(t)).detach().cpu().numpy()

        u_pred_matrix = u_pred.reshape(len(x), len(t)).detach().cpu().numpy()


        # loss_energy = np.max(np.abs(integrate.simpson(u_exact_matrix.T ** 2 - u_pred_matrix.T ** 2, x.cpu().numpy())))
        loss_energy = np.max(np.abs(integrate.romb(u_exact_matrix.T ** 2 - u_pred_matrix.T ** 2, dx = (x[1] - x[0]).cpu().numpy())))
        
        loss = loss_pde + loss_data

        return loss, loss_pde, loss_data, loss_energy

    
    # suitable for LBFGS
    def closure(self):

        # this is more like a not so elegant hack to zero grad both optimizers
        self.optimizer_LBFGS.zero_grad()

        # boundary and initial point
        u_pred_bic = self.model(self.X_bic)

        # the PDE error of inner point
        u_pred = self.model(self.X_train)

        # print(R)

        # compute the loss
        loss_vector = self.loss_fn(u_pred_bic, self.u_bic, u_pred, self.X_train, self.x_train, self.t_train, self.current_epoch)

        loss_vector[0].backward()
        return loss_vector[0]


    def train(self):

        writer = SummaryWriter(comment=f"1DBurgers-Original-NoSolution-bs{self.batch_size}-epoches{self.epoches}-h_train{self.h_train}-tau_train{self.tau_train}-gamma{self.gamma}-width{self.width}-hidden{self.num_hidden}-LBFGS{self.Adam_proportion}-{self.j}")


        size = len(self.train_dataloader.dataset)
        self.model.train()

        for epoch in range(self.epoches):

            if epoch < (self.Adam_proportion * self.epoches):

                for batch, (X_bic, u_bic) in enumerate(self.train_dataloader):
                    # Move the data to the computational device
                    X_bic, u_bic = X_bic.to(self.device), u_bic.to(self.device)

                    # this is more like a not so elegant hack to zero grad both optimizers
                    self.optimizer_adam.zero_grad()

                    # boundary and initial point
                    u_pred_bic = self.model(X_bic)

                    # the PDE error of inner point
                    u_pred = self.model(self.X_train)

                    # print(R)

                    # compute the loss
                    loss_vector = self.loss_fn(u_pred_bic, u_bic, u_pred, self.X_train, self.x_train, self.t_train, epoch)

                    # backpropagation
                    loss_vector[0].backward()
                    self.optimizer_adam.step()
                    self.scheduler.step()

            else:
                for batch, (X_bic, u_bic) in enumerate(self.train_dataloader_LBFGS):
                    # print(X_bic)

                    self.X_bic, self.u_bic = X_bic.to(self.device), u_bic.to(self.device)

                    self.current_epoch = epoch
                
                    self.optimizer_LBFGS.step(self.closure)


                    u_pred_bic = self.model(self.X_bic)

                    # the PDE error of inner point
                    u_pred = self.model(self.X_train)

                    # print(R)

                    # compute the loss
                    loss_vector = self.loss_fn(u_pred_bic, u_bic, u_pred, self.X_train, self.x_train, self.t_train, epoch)

            writer.add_scalar('Train/Total loss', loss_vector[0], epoch)
            writer.add_scalar('Train/PDE loss', loss_vector[1], epoch)
            writer.add_scalar('Train/data loss', loss_vector[2], epoch)
            writer.add_scalar('Train/energy loss', loss_vector[3], epoch)
            writer.add_scalar('Train/shock position loss', loss_vector[-1], epoch)

            writer.add_scalar('Train/Learning Rate',self.optimizer_adam.state_dict()['param_groups'][0]['lr'],epoch)

        if not os.path.exists('./trained_model'):
            os.makedirs('./trained_model')

        torch.save(self.model.state_dict(), f"./trained_model/1DBurgers-Original-NoSolution-bs{self.batch_size}-epoches{self.epoches}-h_train{self.h_train}-tau_train{self.tau_train}-gamma{self.gamma}-width{self.width}-hidden{self.num_hidden}-LBFGS{self.Adam_proportion}-{self.j}.pth")
            

        writer.close()
        
        
    def evalutaion(self):
        h_test = self.h_train / 2
        tau_test = 0.01
        x_test = torch.arange(-1, 1 + h_test, h_test)
        t_test = torch.arange(0, 1 + tau_test, tau_test)

        # lozation of training data
        bc1 = torch.stack(torch.meshgrid(x_test[0], t_test, indexing='ij')).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(x_test[-1], t_test, indexing='ij')).reshape(2, -1).T
        ic = torch.stack(torch.meshgrid(x_test, t_test[0], indexing='ij')).reshape(2, -1).T
        X_test_bic = torch.cat([bc1, bc2, ic]).to(self.device)
        
        
        # the corresponding boundary and initial condition
        u_bc1 = torch.zeros(len(bc1))
        u_bc2 = torch.zeros(len(bc2))
        u_ic = self.Burgers.IC(ic[:, 0])
        u_bic = torch.cat([u_bc1, u_bc2, u_ic]).unsqueeze(1).to(self.device)

        # exact solution
        X_test = torch.stack(torch.meshgrid(x_test, t_test, indexing='ij')).reshape(2, -1).T
        X_test = X_test.to(self.device)
        X_test.requires_grad = True
    
        # self.model.eval()
        # with torch.no_grad():
        u_pred = self.model(X_test)
            
            # boundary and initial point
        u_pred_bic = self.model(X_test_bic)
        
            
        loss_vector = self.loss_fn(u_pred_bic, u_bic, u_pred, X_test, x_test, t_test, self.epoches)

        return torch.tensor(loss_vector)




def parallel_train(j, params, results_queue):

    if j < 5:  # 只运行 5~9
        return
    
    setup_seed(int(2023+ (1000*j)))

    net = structured_PINNs(params, j)
    net.train()

    result = net.evalutaion().cpu()

    # print(result.is_shared())

    results_queue.put((j, result))