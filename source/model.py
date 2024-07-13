import numpy as np
import torch
import scipy
import itertools
from scipy.special import binom
from torch import nn
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
from source.integrators import MonteCarlo
mc = MonteCarlo()


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
        y_in = y.unsqueeze(0).repeat(self.batch_size,1,1)
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
                 smoothing=None):
        super(Leray_Schauder, self).__init__()
        self.basis = basis
        self.epsilon = nn.Parameter(torch.tensor(epsilon, dtype=torch.float32))
        self.dim = dim
        self.integration_domain = integration_domain
        self.channels = channels
        self.N = N
        self.p = p
        self.n = self.basis.basis_size()
        self.batch_size = batch_size
        self.smoothing = smoothing
        
    def norm(self,func):
        integral = mc.integrate(
            fn= lambda s: func(s.to(device))**self.p,
            dim= self.dim,
            N= self.N,
            integration_domain = self.integration_domain,
            out_dim = -2,
            )
        return torch.pow(integral,1/self.p)
        
    def mu_i(self,func,i):
        norm_ = self.norm(lambda s: func(s)-self.basis(i,s)).to(torch.float64)
        return torch.where(norm_<=self.epsilon,self.epsilon-norm_,.001*self.epsilon).float()
        
    def proj(self,func,x):
        out = torch.zeros(self.batch_size,self.channels)
        for i in range(self.n):
            mui = self.mu_i(func,i)
            print(mui.shape)
            out += mui*self.basis.forward(i,x).view(self.batch_size,self.channels)
        out = torch.softmax(out,dim=-1)
        return out
    
    def proj_coeff(self,func):
        out = torch.tensor([]).to(device)
        for i in range(self.n):
            mui = self.mu_i(func,i)
            out = torch.cat([out,mui.unsqueeze(-2)],dim=-2)
        out = torch.softmax(out,dim=-2)
        return out
    
    def basis_eval(self,i,x):
        return self.basis.forward(i,x)
    
    def return_basis(self):
        return self.basis
    
    def return_channels(self):
        return self.channels

    def return_epsilon(self):
        return self.epsilon
    
    
    
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


class multi_linear_interpolate:
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
