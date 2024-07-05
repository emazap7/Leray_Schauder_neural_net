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
        self.shapes = shapes
        self.first = nn.Linear(dim_in,shapes[0])
        self.basis = nn.ModuleList(
                        nn.ModuleList(
                            nn.Linear(shapes[k],shapes[k+1])\
                            for k in range(len(shapes)-1))\
                            for i in range(n))
        self.last = nn.Linear(shapes[-1], dim_out)
        self.NL = NL(inplace=True) 
        self.batch_size = batch_size
    
    def generate_basis(self):
        return self.basis
    
    def forward(self,i,y):
        y_in = y.unsqueeze(0).repeat(self.batch_size,1,1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.basis[i]:
            y = self.NL(layer.forward(y)) 
        y = self.last.forward(y)
        return y
    
    def basis_size(self):
        return self.n
 

    
class Leray_Schauder(nn.Module):
    def __init__(self,basis,epsilon=.1,dim=1,channels=2,N=1000,p=2,batch_size=8):
        super(Leray_Schauder, self).__init__()
        self.basis = basis
        self.epsilon = epsilon
        self.dim = dim
        self.channels = channels
        self.N = N
        self.p = p
        self.n = self.basis.basis_size()
        self.batch_size = batch_size
        
    def norm(self,func):
        integral = mc.integrate(
            fn= lambda s: func(s)**self.p,
            dim= self.dim,
            N= self.N,
            out_dim = -2,
            )
        return torch.pow(integral,1/self.p)
        
    def mu_i(self,func,i):
        norm_ = torch.norm(self.norm(lambda s: func(s)-self.basis(i,s)),p=self.p,dim=[-1]).to(torch.float64)
        norm_ = norm_.unsqueeze(-1)
        return torch.where(norm_<=self.epsilon,norm_,0.).float()#norm_ if norm_<= self.epsilon else 0.
        
    def proj(self,func,x):
        out = torch.zeros(self.batch_size,self.channels)
        normalization = torch.tensor([1e-7]).unsqueeze(0).repeat(self.batch_size,1)
        for i in range(self.n):
            mui = self.mu_i(func,i)
            out += mui*self.basis.forward(i,x).view(self.batch_size,self.channels)
            normalization += mui
        out /= normalization
        return out
    
    def proj_coeff(self,func):
        out = torch.tensor([])
        normalization = torch.tensor([1e-7]).unsqueeze(0).repeat(self.batch_size,1)
        for i in range(self.n):
            mui = self.mu_i(func,i)
            out = torch.cat([out,mui],dim=-1)
            normalization += mui
        out /= normalization
        return out
    
    def basis_eval(self,i,x):
        return self.basis.forward(i,x)
    
    def return_basis(self):
        return self.basis
    
    def return_channels(self):
        return self.channels
    
    
    
class Leray_Schauder_model(nn.Module):
    def __init__(self,LS_map,proj_NN,batch_size=8):
        super(Leray_Schauder_model, self).__init__()
        self.LS_map = LS_map
        self.proj_NN = proj_NN
        self.n = LS_map.return_basis().basis_size()
        self.basis = LS_map.return_basis()
        self.batch_size = batch_size
       
    def reconstruction(self,coeff):
        func = lambda s: (coeff.view(self.batch_size,1,self.n,1)*\
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
            x = time
            y = obs
            coeffs = natural_cubic_spline_coeffs(x, y)
            interpolation = NaturalCubicSpline(coeffs)

            def output(point:torch.Tensor):
                return interpolation.evaluate(point)

            return output
        self.interpolator = interpolator(self,time,obs)
        
    def func(self,x):
        return self.interpolator(x)
