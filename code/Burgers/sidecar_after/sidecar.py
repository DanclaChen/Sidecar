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

import scipy.io

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
    
    
# the subnetwork related to the structure factor R (conservative or dissipative)    
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

        # if width is provided
        if width_R is not None:
            hidden_size = [width_R] * num_hidden_R
        # if hidden_size is provided
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
        layers.append(('output', nn.Linear(hidden_size[-1], output_size, bias = True)))

        layerDict = OrderedDict(layers)
        self.layers = nn.Sequential(layerDict)
        

    def forward(self, t):
        out = self.layers(t) 
        return out
    
    
# the whole network with a output as u = R * v
# to add the regularization term, we have to give the two output separately
class NN(nn.Module):
    def __init__(self, width, num_hidden, width_R, num_hidden_R):
        super(NN, self).__init__()
        self.PDE = PDE_NN(width, num_hidden)
        self.structure = structure_NN(width_R, num_hidden_R)

    def forward(self, X):
        v = self.PDE(X)
        R = self.structure(X[:,1].reshape(-1, 1))
#         print(torch.size(X[:,1]))
        return v, R





class structured_PINNs():
    def __init__(self, params, j, device):

        # multiprocessing index
        self.j = j
        
        # hyperparameters
        self.learning_rate = params.learning_rate
        self.batch_size = params.batch_size
        self.epoches = params.epoches
        self.epoches_read = params.epoches_read
        self.Adam_proportion = params.Adam_proportion
        self.gamma = params.gamma
        self.coe_without_R = params.coe_without_R

        self.coe_structure = params.coe_structure
        self.coe_data = params.coe_data
        self.coe_regular = params.coe_regular

        self.h_train = params.h_train
        self.tau_train = params.tau_train

        self.num_train = params.num_train

        self.tol = params.tol

        # parameters of the Burger's equation
        self.Burgers = Burgers()
        self.nu = self.Burgers.nu
        
        # initialization
        self.device = device

        self.width = params.width
        self.num_hidden = params.num_hidden
        self.width_R = params.width_R
        self.num_hidden_R = params.num_hidden_R

        # the network
        self.model = NN(self.width, self.num_hidden, self.width_R, self.num_hidden_R).to(self.device)

        # load the trained model
        self.model.load_state_dict(torch.load(f"/home/22040517r/sidecar/Sidecar/codes/parallel/simple_largeNu/PINNs_withR/trained_model/1DBurgers-PINNs_withR-bs{self.batch_size}-epoches{self.epoches_read}-h_train{self.h_train}-tau_train{self.tau_train}-gamma{self.gamma}-width{self.width}-hidden{self.num_hidden}-widthR{self.width_R}-hiddenR{self.num_hidden_R}-LBFGS{self.Adam_proportion}-{self.j}.pth"))


        
        
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
        self.train_dataloader = DataLoader(train_set, batch_size = len(train_set), shuffle=True, generator=torch.Generator(device='cuda'))
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
            tolerance_change = 1e-9,
            line_search_fn="strong_wolfe",   # better numerical stability
        )

    
    # the loss function, which contains four parts: PDE, boundary and initial condition, structure factor equation, regularization
    def loss_fn(self, v_pred_bic, R_bic, u_bic, v_pred, R, X, x, t, epoch):
    
        u_pred_bic = v_pred_bic * R_bic
        u_pred = v_pred * R
        
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
        

####### 3. structure error
        # the Burgers function satisfies the dissipative law as 
        # d/dt \int u^2 dx = -2 \nu \int u_x^2 dx
        
        # x_temp = torch.arange(-1, 1, self.h)

        # v_pred_matrix = v_pred.reshape(len(x_temp), len(t)).cpu().numpy()

        # find the shock to divide the integral
        du_dx_matrix = du_dx.reshape(len(x), len(t)).detach().cpu().numpy()
        x_shock = np.argmax(np.abs(du_dx_matrix[:, -1]))

        v_pred_matrix = v_pred.reshape(len(x), len(t)).detach().cpu().numpy()

        h = x[1] - x[0]
        theta_1 = integrate.romb(v_pred_matrix.T ** 2, dx = h.cpu().numpy(), axis = 1)

        # print(np.shape(theta_1[:2]))
        
        dv_dX = torch.autograd.grad(
                inputs = X, 
                outputs = v_pred, 
                grad_outputs = torch.ones_like(v_pred), 
                retain_graph = True, 
                # create_graph = True
            )[0]
        
        dv_dx_matrix = dv_dX[:, 0].reshape(len(x), len(t)).detach().cpu().numpy()

        theta_2 = integrate.romb(dv_dx_matrix.T ** 2, dx = h.cpu().numpy(), axis = 1)

        RHS_short = -2 * self.nu * torch.from_numpy(theta_2).reshape(-1, 1).to(self.device) * (R**2)[:len(t)]

        LHS_temp_short = torch.from_numpy(theta_1).to(self.device).reshape(-1, 1) * (R**2)[:len(t)]

        loss_structure_vector_short = torch.zeros(len(t)).to(self.device)

        # ideal IC f(x) = IC(x)
        # R_0 = \int dx u_0^2(x) / \int dx v^2(x,0)

        u_0 = self.Burgers.IC(x).cpu().numpy()

        Q_0 = integrate.romb(u_0**2, dx = h.cpu().numpy())

        loss_structure_vector_short[0] = (LHS_temp_short[0] - Q_0)**2

        for i in range(1, len(t)):
            loss_structure_vector_short[i] = (LHS_temp_short[i] - LHS_temp_short[i-1] - self.tau_train * RHS_short[i])**2

        # introduce causality of loss function
        M_sum = torch.triu(torch.ones(len(t), len(t)), diagonal=1).T
        with torch.no_grad():
            W = torch.exp(-self.tol * torch.matmul(M_sum, loss_structure_vector_short))

        loss_structure_vector_causal = W * loss_structure_vector_short

        loss_structure = torch.mean(loss_structure_vector_causal)



        u_exact_matrix = self.Burgers.exact_solution(X).reshape(len(x), len(t)).detach().cpu().numpy()

        u_pred_matrix = u_pred.reshape(len(x), len(t)).detach().cpu().numpy()

        energy_numerical = integrate.romb(u_pred_matrix.T ** 2, dx = h.cpu().numpy(), axis = 1)
        energy_exact = integrate.romb(u_exact_matrix.T ** 2, dx = h.cpu().numpy(), axis = 1)


        loss_energy = np.max(np.abs(energy_numerical - energy_exact))


        # to measure the postion error of shock waves
        loss_shock_position = torch.abs(x[x_shock] - x[int(len(x) / 2)])



####### 5. regularization error
        
        # loss_regular = self.criterion(R, torch.ones(R.size()).to(self.device))
        loss_regular = 0.0

        loss_exact = self.criterion(u_pred.squeeze(), self.Burgers.exact_solution(X)) 

        # total error

        # To ensure that the PDE network can learn enough things to guide the R
        if epoch < self.coe_without_R * self.epoches:
            loss = loss_pde + self.coe_data * loss_data
            # print(f"without R{epoch}")
        else:
            loss = loss_pde + self.coe_data * loss_data + self.coe_structure * loss_structure + self.coe_regular * loss_regular



        return loss, loss_pde, loss_data, loss_exact, loss_structure, loss_energy, loss_shock_position

    
    # suitable for LBFGS
    def closure(self):

        # Move the data to the computational device
        # X_bic, u_bic = X_bic.to(self.device), u_bic.to(self.device)

        # this is more like a not so elegant hack to zero grad both optimizers
        self.optimizer_LBFGS.zero_grad()

        # boundary and initial point
        v_pred_bic, R_bic = self.model(self.X_bic)

        # the PDE error of inner point
        v_pred, R = self.model(self.X_train)

        # print(R)

        # compute the loss
        loss_vector = self.loss_fn(v_pred_bic, R_bic, self.u_bic, v_pred, R, self.X_train, self.x_train, self.t_train, self.current_epoch)

        loss_vector[0].backward()
        return loss_vector[0]


    def train(self):   



        writer = SummaryWriter(comment=f"1DBurgers-noR{self.coe_without_R}-bs{self.batch_size}-epoches{self.epoches}-h_train{self.h_train}-tau_train{self.tau_train}-gamma{self.gamma}-coe_structure{self.coe_structure}-coe_data{self.coe_data}-{self.j}")


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
                    v_pred_bic, R_bic = self.model(X_bic)

                    # the PDE error of inner point
                    v_pred, R = self.model(self.X_train)

                    # compute the loss
                    loss_vector = self.loss_fn(v_pred_bic, R_bic, u_bic, v_pred, R, self.X_train, self.x_train, self.t_train, epoch)


                    
                    # backpropagation
                    loss_vector[0].backward()
                    self.optimizer_adam.step()
                    self.scheduler.step()

            else:
                for batch, (X_bic, u_bic) in enumerate(self.train_dataloader_LBFGS):


                    self.current_epoch = epoch

                    self.X_bic, self.u_bic = X_bic.to(self.device), u_bic.to(self.device)
                
                # self.optimizer_LBFGS.zero_grad()
                
                self.optimizer_LBFGS.step(self.closure)

                v_pred_bic, R_bic = self.model(self.X_bic)

                # the PDE error of inner point
                v_pred, R = self.model(self.X_train)

                # compute the loss
                loss_vector = self.loss_fn(v_pred_bic, R_bic, u_bic, v_pred, R, self.X_train, self.x_train, self.t_train, epoch)



            writer.add_scalar('Train/Total loss', loss_vector[0], epoch)
            writer.add_scalar('Train/PDE loss', loss_vector[1], epoch)
            writer.add_scalar('Train/data loss', loss_vector[2], epoch)
            writer.add_scalar('Train/regularization loss', loss_vector[3], epoch)
            writer.add_scalar('Train/structure loss', loss_vector[4], epoch)
            writer.add_scalar('Train/energy loss', loss_vector[5], epoch)
            writer.add_scalar('Train/shock position loss', loss_vector[-1], epoch)

            writer.add_scalar('Train/Learning Rate',self.optimizer_adam.state_dict()['param_groups'][0]['lr'],epoch)

        if not os.path.exists('./trained_model'):
            os.makedirs('./trained_model')

        torch.save(self.model.state_dict(), f"./trained_model/1DBurgers-R0_square-noR{self.coe_without_R}-bs{self.batch_size}-epoches{self.epoches}-epoches_read{self.epoches_read}-h_train{self.h_train}-tau_train{self.tau_train}-gamma{self.gamma}-coe_structure{self.coe_structure}-coe_data{self.coe_data}-width{self.width}-hidden{self.num_hidden}-widthR{self.width_R}-hidden_R{self.num_hidden_R}-LBFGS{self.Adam_proportion}-lr{self.learning_rate}-{self.j}.pth")

        writer.close()
        
        
    def evaluation(self):
        h_test = self.h_train / 2
        tau_test = self.tau_train / 2
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
        v_pred, R = self.model(X_test)

        # print(R)
            
            # boundary and initial point
        v_pred_bic, R_bic = self.model(X_test_bic)

            
        loss_vector = self.loss_fn(v_pred_bic, R_bic, u_bic, v_pred, R, X_test, x_test, t_test, self.epoches)

        self.model.zero_grad()


        return torch.tensor(loss_vector)




def parallel_train(j, params, results_queue, device):
    # Set the default data type to float64
    torch.set_default_dtype(torch.float64)
    # Set the device for this process
    torch.cuda.set_device(device)

    setup_seed(int(2023 + (1000 * j)))

    net = structured_PINNs(params, j, device)

    loss_before = net.evaluation().to('cpu')
    # print(hasattr(net, 'evaluation'))
    # net.to(device)  # Move the network to the specified device
    net.train()

    loss_after = net.evaluation().to('cpu')

    result = torch.cat([loss_after, loss_before])  # Ensure result is on CPU before putting it in the queue

    results_queue.put((j, result))

