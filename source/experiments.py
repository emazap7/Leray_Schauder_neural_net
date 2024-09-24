import logging
import warnings
from typing import Callable, Optional, Union
import os, argparse

import numpy as np
import torch

from scipy import integrate
import time
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook as tqdm
from torch.nn  import functional as F

logger = logging.getLogger("iesolver")
logger.setLevel(logging.WARNING)#(logging.DEBUG)

import matplotlib.pyplot as plt


import torchcubicspline
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
                             
#from torchdiffeq import odeint


from source.integrators import MonteCarlo 
mc = MonteCarlo()

from source.model import F_NN, basis, Leray_Schauder, Leray_Schauder_model, interpolated_func, multidim_interpolate
from source.utils import dataset, burgers_dataset, EarlyStopping, SaveBestModel, load_checkpoint

from source.utils import fix_random_seeds,to_np

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"



def experiment(model, Data, time_seq, args):
    
    
    
    str_model_name = "Leray_Schauder"
    
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    obs = Data
    times = time_seq

    decoder = args.decoder
    
    All_parameters = list(model.parameters()) + list(decoder.parameters())
     
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        model, optimizer, scheduler, _, decoder = load_checkpoint(path, model, optimizer, scheduler, None, decoder)
        
    
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]

        save_best_model = SaveBestModel()
        
        
        split_size = int(args.training_split*obs.size(0))
        
        Dataset_train = dataset(Data[:obs.shape[0]-split_size,...],args.downsample)
        Dataset_valid = dataset(Data[obs.shape[0]-split_size:,...],args.downsample)
        train_loader = DataLoader(Dataset_train,batch_size=args.n_batch,shuffle=True,drop_last=True)
        valid_loader = DataLoader(Dataset_valid,batch_size=args.n_batch,shuffle=False,drop_last=True)

        
        start = time.time()
        for i in range(args.epochs):
            
            model.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
            
            for  inputs_, obs_ in tqdm(train_loader): 
                obs_func_ = interpolated_func(times[::args.downsample],inputs_)
                obs_func = lambda s: obs_func_.func(s.reshape(args.N_MC)).repeat(1,1,int(args.channels/inputs_.shape[-1]))
                if args.plot_train and i%args.plot_freq==0:
                    times_ = torch.linspace(-1,1,args.N_MC).to(device)
                    plt.figure(1, figsize=(8,8),facecolor='w')
                    plt.plot(obs_func(times_).cpu()[0,...,0],obs_func(times_).cpu()[0,...,1],label='Initialization')
                    plt.show()
                    plt.close('all')
                func_ = model.projected_function(obs_func)
                z_ = decoder(func_(times.unsqueeze(-1)))
                
                loss = F.mse_loss(z_, obs_)
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  split_size==0 and args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, z_, inputs_

            ## Validating
                
            model.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                if split_size>0:
                    
                    for  inputs_val, obs_val in tqdm(valid_loader): 
                
                        obs_func_val_ = interpolated_func(times[::args.downsample],inputs_val)
                        obs_func_val = lambda s: \
                        obs_func_val_.func(s.reshape(args.N_MC)).repeat(1,1,int(args.channels/inputs_val.shape[-1]))
                        
                        func_val = model.projected_function(obs_func_val)
                        z_val = decoder(func_val(times.unsqueeze(-1)))
                        
                        loss_validation = F.mse_loss(z_val, obs_val)
                        
                        if i % args.plot_freq == 0:
                            z_p = z_val
                            z_p = to_np(z_p)

                            obs_print = to_np(obs_val[0,...])

                            plt.figure(1, figsize=(8,8),facecolor='w')

                            plt.scatter(obs_print[:,0],obs_print[:,1],label='Data')
                            plt.plot(z_p[0,:,0],z_p[0,:,1],label='Model')
                            plt.savefig(os.path.join(path_to_save_plots,'plot_'+str(i)))
                            plt.close('all')
                                
                            del z_p, obs_print
                        
                        del obs_val, z_val, inputs_val

                        counter += 1
                        val_loss += loss_validation.item()
                        
                        del loss_validation

                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)
                
                
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            if i % args.plot_freq == 0:

                plt.figure(0, figsize=(8,8),facecolor='w')


                plt.plot(np.log10(all_train_loss),label='Train loss')
                if split_size>0:
                    plt.plot(np.log10(all_val_loss),label='Val loss')
                plt.xlabel("Epoch")
                plt.ylabel("MSE Loss")

                plt.savefig(os.path.join(path_to_save_plots,'losses'))
                        

            end_i = time.time()
            #print("Time epoch"+str(i)+": ", end_i-start_i)

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if split_size>0:
                save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, None, decoder)
            else:
                save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, model, None, decoder)


            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
        tot_time = end - start
        f = open('time_results.txt','w')
        f.write(str(tot_time) + '\n')
        f.close()

        return tot_time
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        
        Dataset_test = dataset(Data,args.downsample)
        test_loader = DataLoader(Dataset_test,batch_size=1,shuffle=False)
        
        
        ## Validating
        model.eval()
        decoder.eval()

        test_loss = 0.0
        all_test_loss=[]
        counter = 0
        
        for  inputs_test, obs_test in tqdm(test_loader): 
                
            obs_func_test_ = interpolated_func(times[::args.downsample],inputs_test)
            obs_func_test = lambda s: \
            obs_func_test_.func(s.reshape(args.N_MC)).repeat(1,1,int(args.channels/inputs_test.shape[-1]))
            func_test = model.projected_function(obs_func_test)
            z_test = decoder(func_test(times.unsqueeze(-1)))

            loss_test = F.mse_loss(z_test, obs_test)
            
            
            if args.dataset_name == 'fMRI':
                z_p = to_np(z_test[0,...])
                z_p = gaussian_filter(z_p,sigma=3)
                obs_p = to_np(obs_test[0,...])
                obs_p = gaussian_filter(obs_p,sigma=3)
                
                plt.figure(counter, figsize=(8,8),facecolor='w')
                plt.imshow(z_p.transpose(0,1))
                plt.show()
                plt.figure(counter+len(test_loader), figsize=(8,8),facecolor='w')
                plt.imshow(obs_p.transpose(0,1))
                plt.show()
            
            del obs_test, z_test, inputs_test

            counter += 1
            test_loss += loss_test.item()
            all_test_loss.append(loss_test.item())
            
            del loss_test
        

        test_loss /= counter

        return test_loss, torch.tensor(all_test_loss).std()





def burgers_experiment(model, Data, encoder, decoder, spacetime_domain_tensor, args):
    
    
    
    str_model_name = "Leray_Schauder"
    
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    obs = Data
    spacetime_domain = spacetime_domain_tensor
    
    
    All_parameters = list(model.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
     
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        model, optimizer, scheduler, encoder, decoder = load_checkpoint(path, model, optimizer, scheduler, encoder, decoder)
        
    
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]
        epsilon_vals = []

        save_best_model = SaveBestModel()
        
        
        split_size = int(args.training_split*obs.size(0))
        
        Dataset_train = burgers_dataset(Data[:obs.shape[0]-split_size,...],type='BV')
        Dataset_valid = burgers_dataset(Data[obs.shape[0]-split_size:,...],type='BV')
        train_loader = DataLoader(Dataset_train,batch_size=args.n_batch,shuffle=True,drop_last=True)
        valid_loader = DataLoader(Dataset_valid,batch_size=args.n_batch,shuffle=True,drop_last=True)

        
        start = time.time()
        for i in range(args.epochs):
            
            model.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
            
            for  inputs_, obs_ in tqdm(train_loader): 
                obs_func = lambda s:\
                            encoder(torch.nn.functional.interpolate(inputs_,[args.burgers_t],mode='linear').unsqueeze(-1))
                func_ = model.projected_function(obs_func)
                z_ = func_(spacetime_domain).view(args.n_batch,args.n_points,args.burgers_t,args.channels)
                z_ = decoder(z_)
                loss = F.mse_loss(z_.squeeze(-1), obs_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  split_size==0 and args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, z_, inputs_

            ## Validating
                
            model.eval()
            encoder.eval()
            decoder.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                if split_size>0:
                    
                    for  inputs_, obs_ in tqdm(valid_loader): 
                        obs_func = lambda s:\
                                    encoder(torch.nn.functional.interpolate(inputs_,[args.burgers_t],mode='linear').unsqueeze(-1))
                        func_ = model.projected_function(obs_func)
                        z_ = func_(spacetime_domain).view(args.n_batch,args.n_points,args.burgers_t,args.channels)
                        z_ = decoder(z_)
                        loss = F.mse_loss(z_.squeeze(-1), obs_)
                        
                        
                        if i % args.plot_freq == 0:
                            z_p = z_
                            z_p = to_np(z_p)

                            obs_print = to_np(obs_[0,...])

                            plt.figure(1, figsize=(8,8),facecolor='w')
                            plt.imshow(obs_print,aspect='auto')
                            plt.savefig(os.path.join(path_to_save_plots,'plot_obs'+str(i)))
                            plt.figure(2, figsize=(8,8),facecolor='w')
                            plt.imshow(z_p[0,...],aspect='auto')
                            plt.savefig(os.path.join(path_to_save_plots,'plot_pred'+str(i)))
                            plt.close('all')
                                
                            del z_p, obs_print
                        
                        del obs_, z_, inputs_

                        counter += 1
                        val_loss += loss.item()
                        
                        del loss

                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)
                
                
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)
                if args.train_epsilon:
                    epsilon_vals.append(to_np(model.LS_map.return_epsilon()))
                
                del val_loss

            if i % args.plot_freq == 0:

                plt.figure(0, figsize=(8,8),facecolor='w')


                plt.plot(np.log10(all_train_loss),label='Train loss')
                if split_size>0:
                    plt.plot(np.log10(all_val_loss),label='Val loss')
                plt.xlabel("Epoch")
                plt.ylabel("MSE Loss")

                plt.savefig(os.path.join(path_to_save_plots,'losses'))
                
                if args.train_epsilon:
                    plt.figure(3, figsize=(8,8),facecolor='w')
                    plt.plot(epsilon_vals)
                    plt.xlabel("Epoch")
                    plt.ylabel("Epsilon")
    
                    plt.savefig(os.path.join(path_to_save_plots,'epsilon'))
                        

            end_i = time.time()
            #print("Time epoch"+str(i)+": ", end_i-start_i)

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if split_size>0:
                save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, encoder, decoder)
            else:
                save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, model, encoder, decoder)


            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
        tot_time = end - start
        f = open('time_results.txt','w')
        f.write(str(tot_time) + '\n')
        f.close()

        return tot_time
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        path_to_save_plots = os.path.join(path_to_experiment,'test_'+str(args.resume_from_checkpoint),'plots_test')
        os.makedirs(path_to_save_plots)
        Dataset_test = burgers_dataset(Data,type='BV')
        test_loader = DataLoader(Dataset_test,batch_size=1,shuffle=False,drop_last=False)
        
        
        ## Validating
        model.eval()
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            test_loss = 0.0
            all_test_loss=[]
            counter = 0
            
            for  inputs_, obs_ in tqdm(test_loader): 
                    
                obs_func = lambda s:\
                            encoder(torch.nn.functional.interpolate(inputs_,[args.burgers_t],mode='linear').unsqueeze(-1))
                func_ = model.projected_function(obs_func)
                z_ = func_(spacetime_domain).view(args.n_batch,args.n_points,args.burgers_t,args.channels)
                z_ = decoder(z_)
                loss_test = F.mse_loss(z_.squeeze(-1), obs_)
    
                counter += 1
                test_loss += loss_test.item()
                all_test_loss.append(loss_test.item())

                z_p = z_
                z_p = to_np(z_p)

                obs_print = to_np(obs_[0,...])

                plt.figure(1, figsize=(8,8),facecolor='w')
                plt.imshow(obs_print,aspect='auto')
                plt.savefig(os.path.join(path_to_save_plots,'plot_obs_test'+str(counter)))
                plt.figure(2, figsize=(8,8),facecolor='w')
                plt.imshow(z_p[0,...],aspect='auto')
                plt.savefig(os.path.join(path_to_save_plots,'plot_pred_test'+str(counter)))
                plt.close('all')
                
                del loss_test, obs_, z_, inputs_
    
            test_loss /= counter
    
            return test_loss, torch.tensor(all_test_loss).std()
