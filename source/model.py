import numpy as np
import torch
import scipy
import itertools
from scipy.special import binom
from torch import nn
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
from source.integrators import MonteCarlo, IntegrationGrid
mc = MonteCarlo()

from source.Galerkin_transformer import SimpleTransformerEncoderLayer, SimpleTransformerEncoderLastLayer


if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"


class F_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(F_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y
        y = y_in.flatten(-2,-1)
        y = self.NL(self.first.forward(y))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)
        return y.view(y_in.shape[0],y_in.shape[1],y_in.shape[2])

class Simple_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(Simple_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y
        y = self.NL(self.first.forward(y))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)
        return y
    
    
class basis(nn.Module):
    def __init__(self,dim_in,dim_out,n=8,shapes=[16,16],NL=nn.ELU,batch_size=8):
        super(basis, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n = n
        self.first = nn.ModuleList(nn.Linear(dim_in,shapes[0]) for i in range(n))
        self.shapes = shapes
        self.basis = nn.ModuleList(
                        nn.ModuleList(
                            nn.Linear(self.shapes[k],self.shapes[k+1])\
                            for k in range(len(self.shapes)-1))\
                            for i in range(n))
        self.last = nn.ModuleList(nn.Linear(shapes[-1],dim_out) for i in range(n))
        self.NL = NL(inplace=True) 
        self.batch_size = batch_size
    
    def generate_basis(self):
        return self.basis
    
    def forward(self,i,y):
        y_in = y.unsqueeze(0).repeat([self.batch_size]+[1 for i in range(len(y.shape))])
        y = self.first[i](y_in)
        for layer in self.basis[i]:
            y = self.NL(layer.forward(y)) 
        y = self.last[i](y)
        return y
    
    def basis_size(self):
        return self.n
 

    
class Leray_Schauder(nn.Module):
    def __init__(self,basis,
                 epsilon=.1,
                 dim=1,
                 integration_domain = [[-1,1]],
                 channels=2,
                 N=1000,
                 p=2,
                 batch_size=8,
                 norm_type='Lp',
                 norm_nn = None,
                 mui = None,
                 train_epsilon=True,
                 grid = None,
                 softmax = True):
        super(Leray_Schauder, self).__init__()
        self.basis = basis
        if train_epsilon==True:
            self.epsilon = nn.Parameter(torch.tensor(epsilon, dtype=torch.float32)).to(device)
        else:
            self.epsilon = epsilon
        self.dim = dim
        self.integration_domain = integration_domain
        self.channels = channels
        self.N = N
        self.p = p
        self.n = self.basis.basis_size()
        self.batch_size = batch_size
        self.norm_type = norm_type
        self.norm_nn = norm_nn
        self.mui = mui
        if (norm_type=='LP') == False:
            if grid == None:
                grid = IntegrationGrid(N,integration_domain)
                self.points = grid.return_grid().to(device)
            else:
                self.points = grid
        self.softmax = softmax
        
    def norm(self,func):
        if self.norm_nn == None:
            integral = mc.integrate(
                fn= lambda s: torch.abs(func(s.to(device)))**self.p,
                dim= self.dim,
                N= self.N,
                integration_domain = self.integration_domain,
                out_dim = -2,
                )
            return torch.pow(integral,1/self.p)
        else:
            integral = mc.integrate(
                fn= lambda s: torch.abs(self.norm_nn(s.to(device))*func(s.to(device)))**self.p,
                dim= self.dim,
                N= self.N,
                integration_domain = self.integration_domain,
                out_dim = -2,
                )
            return torch.pow(integral,1/self.p)
        
    def mu_i(self,func,i):
        if self.norm_type=='Lp':
            norm_ = self.norm(lambda s: func(s)-self.basis(i,s)).to(torch.float64)
            return torch.where(norm_<=self.epsilon,self.epsilon-norm_,.001*self.epsilon).float()
        else:
            if self.mui == None:
                basis_eval = self.basis(i,self.points)
                out = func(self.points).view_as(basis_eval) - basis_eval
                norm_ = torch.linalg.vector_norm(out,dim=[1],ord=self.p)
                return torch.where(norm_<=self.epsilon,self.epsilon-norm_,.001*self.epsilon).float()
            else:
                y1 = func(self.points)
                y2 = self.basis(i,self.points).view_as(y1)
                out = self.mui(y1 - y2)
                return out
        
    def proj(self,func,x):
        out = torch.zeros(self.batch_size,self.channels)
        for i in range(self.n):
            mui = self.mu_i(func,i)
            out += mui*self.basis.forward(i,x).view(self.batch_size,self.channels)
        out = torch.softmax(out,dim=-1)
        return out
    
    def proj_coeff(self,func):
        out = torch.tensor([]).to(device)
        for i in range(self.n):
            mui = self.mu_i(func,i)
            out = torch.cat([out,mui.unsqueeze(-2)],dim=-2)
        if self.softmax:
            out = torch.softmax(out,dim=-2)
        return out
    
    def basis_eval(self,i,x):
        return self.basis.forward(i,x)
    
    def return_basis(self):
        return self.basis
    
    def return_channels(self):
        return self.channels

    def return_epsilon(self):
        return torch.abs(self.epsilon)

    def return_points(self):
        return self.points
    
    
class Leray_Schauder_model(nn.Module):
    def __init__(self,LS_map,proj_NN,batch_size=8):
        super(Leray_Schauder_model, self).__init__()
        self.LS_map = LS_map
        self.proj_NN = proj_NN
        self.n = LS_map.return_basis().basis_size()
        self.basis = LS_map.return_basis()
        self.batch_size = batch_size
       
    def reconstruction(self,coeff):
        func = lambda s: (coeff.view(self.batch_size,1,self.n,self.LS_map.return_channels())*\
                torch.cat([
                self.LS_map.basis_eval(i,s).unsqueeze(-2)\
                for i in range(self.n)],dim=-2)).sum(dim=-2)
        return func
    
    def projected_function(self,func):
        projection_coeff = self.LS_map.proj_coeff(func)
        out = self.proj_NN.forward(projection_coeff)
        out_func = self.reconstruction(out)
        return out_func

    def return_LS(self):
        return self.LS_map
    
    
    
class interpolated_func:
    def __init__(self,time,obs):
        def interpolator(self,time,obs):
            x = time.to(device)
            y = obs.to(device)
            coeffs = natural_cubic_spline_coeffs(x, y)
            interpolation = NaturalCubicSpline(coeffs)

            def output(point:torch.Tensor):
                return interpolation.evaluate(point)
            return output
        self.interpolator = interpolator(self,time,obs)
        
    def func(self,x):
        return self.interpolator(x)


class multidim_interpolate:
    def __init__(self,x_points, y_points):
        sorted_indices = torch.argsort(x_points)
        self.x_points = x_points[sorted_indices]
        self.y_points = y_points[:,sorted_indices,:]

    def interpolate(self,x):
        y = torch.tensor([]).to(device)
        for i, xi in enumerate(x):
            for j in range(self.x_points.shape[0] - 1):
                if self.x_points[j] <= xi <= self.x_points[j + 1]:
                    x0, y0 = self.x_points[j], self.y_points[:,j,:].unsqueeze(-2)
                    x1, y1 = self.x_points[j + 1], self.y_points[:,j + 1,:].unsqueeze(-2)
                    yi = y0 + ((y1 - y0) / (x1 - x0)) * (xi - x0)
                    y = torch.cat([y,yi],dim=-2)
        return y

class custom_mui(nn.Module):
    def __init__(self,in_dim,channels=8,K1=[(16,2),(16,2)],K2=[(16,2),(16,2)],hidden_dim=32,hidden_ff=64,NL=nn.ELU):
        super(custom_mui, self).__init__()
        self.in_dim = in_dim
        self.channels = channels
        self.K1 = K1
        self.K2 = K2
        self.conv_layer1 = nn.Conv2d(in_dim, hidden_dim,
                                     kernel_size=K1[0],
                                     stride=K1[1])
        self.conv_layer2 = nn.Conv2d(hidden_dim, hidden_dim,
                                     kernel_size=K2[0],
                                     stride=K2[1])
        
        self.NL = NL(inplace=True) 
        self.fc1 = nn.Linear(hidden_ff,hidden_ff)
        self.fc2 = nn.Linear(hidden_ff, channels)

    def forward(self,x):
        x = x.permute(0,3,1,2)
        out = self.NL(self.conv_layer1(x))
        out = self.NL(self.conv_layer2(out))
        out = out.flatten(start_dim=1,end_dim=-1)
        out = self.fc1(out)
        out = self.NL(out)
        out = self.fc2(out)

        return out


class custom_mui1D(nn.Module):
    def __init__(self,in_dim,channels=80,K1=[(4),(4)],K2=[(4),(4)],hidden_dim=32,hidden_ff=32,NL=nn.ELU):
        super(custom_mui1D, self).__init__()
        self.in_dim = in_dim
        self.channels = channels
        self.K1 = K1
        self.K2 = K2
        self.conv_layer1 = nn.Conv1d(in_dim, hidden_dim,
                                     kernel_size=K1[0],
                                     stride=K1[1])
        self.conv_layer2 = nn.Conv1d(hidden_dim, hidden_dim,
                                     kernel_size=K2[0],
                                     stride=K2[1])
        
        self.NL = NL(inplace=True) 
        self.fc1 = nn.Linear(hidden_ff,hidden_ff)
        self.fc2 = nn.Linear(hidden_ff, channels)

    def forward(self,x):
        x = x.permute(0,2,1)
        out = self.NL(self.conv_layer1(x))
        out = self.NL(self.conv_layer2(out))
        out = out.flatten(start_dim=1,end_dim=-1)
        out = self.fc1(out)
        out = self.NL(out)
        out = self.fc2(out)

        return out


class ConvNeuralNet1D(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, out_dim=32,hidden_ff=256,Data_shape1=256,n_patch=32):
        super(ConvNeuralNet1D, self).__init__()
        self.conv_layer1 = nn.Conv1d(dim, hidden_dim,
                                     kernel_size=[int(Data_shape1/n_patch/2)],
                                     stride=int(Data_shape1/n_patch/2))
        
        #self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        if int(Data_shape1/n_patch/4)>1:
            self.conv_layer2 = nn.Conv1d(hidden_dim, out_dim,
                                         kernel_size=[int(Data_shape1/n_patch/4)],
                                         stride=int(Data_shape1/n_patch/4))
        else:
            self.conv_layer2 = nn.Conv1d(hidden_dim, out_dim,
                                         kernel_size=[int(Data_shape1/n_patch/2)],
                                         stride=int(Data_shape1/n_patch/2))
        
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        #print('conv1:',out.shape)
        
        #out = self.max_pool1(out)
        #print('pool1:',out.shape)
        
        out = self.conv_layer2(out)
        #print('conv2:',out.shape)
        
        #out = self.max_pool2(out)
        #print('pool2:',out.shape)
        
        out = out.permute(0,2,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.permute(0,2,1)
        return out 


class ConvNeuralNet(nn.Module):
    def __init__(self, dim, hidden_cnn=32, hidden_dim = 32, out_dim=32,hidden_ff=64,kernels=[[16,2],[16,2]],strides=[[8,2],[8,2]]):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(dim, hidden_cnn,
                                         kernel_size=kernels[0],
                                         stride=strides[0])
        
        self.conv_layer2 = nn.Conv2d(hidden_cnn, hidden_cnn,
                                     kernel_size=kernels[1],
                                     stride=strides[1])
        
        self.fc1 = nn.Linear(hidden_cnn, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.permute(0,2,3,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.permute(0,3,1,2)
        return out   


class Decoder_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(Decoder_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y.permute(0,2,1,3)
        y_in = y_in.flatten(2,3)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)
        y = y_out.permute(0,2,1)

        return y


class model_blocks(nn.Module):
    def __init__(self,dimension,dim_emb,n_head, n_blocks,n_ff, attention_type, dim_out=2, Final_block = False,dropout=0.1,lower_bound=None,upper_bound=None):
        super(model_blocks, self).__init__()
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.first = nn.Linear(dimension,dim_emb)
        self.blocks = nn.ModuleList([SimpleTransformerEncoderLayer(
                                 d_model=dim_emb,n_head=n_head,
                                 dim_feedforward=n_ff,
                                 attention_type=attention_type,
                                 dropout=dropout) for i in range(n_blocks)])
        self.last_block = nn.Linear(dim_emb,dimension)
        
    def forward(self, x, dynamical_mask=None):
        
        x = self.first.forward(x)
        for block in self.blocks:
            x = block.forward(x,dynamical_mask=dynamical_mask) 
        x = self.last_block.forward(x)

        return x
