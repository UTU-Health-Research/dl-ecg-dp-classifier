import os, sys
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from .models.seresnet18 import resnet18
from .models.seresnet18_v2 import resnet18_v2
from ..dataloader.dataset import ECGDataset, get_transforms
from .metrics import cal_multilabel_metrics, roc_curves
import pickle
from opacus import PrivacyEngine
import wandb
import matplotlib.pyplot as plt
import logging
from opacus.utils.batch_memory_manager import BatchMemoryManager


logging.basicConfig(level=logging.INFO)

class Training(object):
    
    def __init__(self, args,sweep_config):
        self.args = args
        self.sweep_config = sweep_config
        
        
    def setup(self):
        '''Initializing the device conditions, datasets, dataloaders, 
        model, loss, criterion and optimizer
        '''

        # Consider the GPU or CPU condition
        if torch.cuda.is_available():
            gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "1")  # Get the GPU index from CUDA_VISIBLE_DEVICES
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
            self.device = torch.device("cuda")
            self.device_count = self.args.device_count
            self.args.logger.info('using gpu device {}'.format(gpu_index))
            # assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            self.args.logger.info('using {} cpu'.format(self.device_count))
        
        self.max_physical_batch_size = self.args.max_physical_batch_size
    

    def train_with_sweep(self):
        self.sweep_id = wandb.sweep(self.sweep_config, project= self.args.project_name)
        wandb.agent(self.sweep_id,self.train)


    def train(self, config= None):
        ''' PyTorch training loop
        '''
        with wandb.init(config = None): 
            config = wandb.config  

            # Check if the logical batch size (origanal batch size) is big them the memory will crash and we need to split it down into samller physical batches
            if config.batch_size < 400: 
                self.args.max_physical_batch_size = config.batch_size
                print('max_physical_batch_size is : ', self.max_physical_batch_size)

            # Creating DataLoader object for the train samples
            training_set = ECGDataset(self.args.train_path, get_transforms('train'), amount = None)
            channels = training_set.channels
            self.train_dl = DataLoader(training_set,
                                    batch_size= config.batch_size,
                                    shuffle=True,
                                    num_workers=self.args.num_workers,
                                    pin_memory=(True if self.device == 'cuda' else False),
                                    drop_last=False)

            self.args.logger.info('total training samples is {}'.format(len(training_set)))

            # defining the resnet model latest version
            self.model = resnet18_v2(in_channel=channels, 
                                out_channel=len(self.args.labels))

            # If more than 1 CUDA device used, use data parallelism
            if self.device_count > 1:
                self.model = torch.nn.DataParallel(self.model) 
            
            # Optimizer
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=self.args.lr,
                                        weight_decay=self.args.weight_decay)

            self.delta =  1/len(self.train_dl)

            ''' 
                 Here we define the Privacy Engine we want to add to the model
            '''
            if (self.args.with_privacy): ## we want privacy
                self.args.logger.info('Training with DP-SGD algorithm')
                self.args.logger.info('Training DP with accountant{}'.format(self.args.accountant))
                
                self.privacy_engine = PrivacyEngine(accountant=self.args.accountant)
                self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_dl,
                    target_epsilon= config.epsilon,
                    target_delta= self.delta,
                    epochs = config.epochs,
                    max_grad_norm = config.max_grad_norm
                    )
        
                print(f"Noise: {self.optimizer.noise_multiplier}")
                
            self.criterion = nn.BCEWithLogitsLoss()
            self.sigmoid = nn.Sigmoid()
            self.sigmoid.to(self.device)
            self.model.to(self.device)
  
         
            self.args.logger.info('train() called: model={}, opt={}(lr={}), epochs={}, device={}'.format(
                    type(self.model).__name__, 
                    type(self.optimizer).__name__,
                    self.optimizer.param_groups[0]['lr'], 
                    self.args.epochs, 
                    self.device))
        
        
            for epoch in range(1, config.epochs+1):
                # --- TRAIN ON TRAINING SET ------------------------------------------
                self.model.train()            
                train_loss = 0.0
                labels_all = torch.tensor((), device=self.device) # , device=torch.device('cuda:0')
                logits_prob_all = torch.tensor((), device=self.device)
                
                batch_loss = 0.0
                batch_count = 0
                step = 0

                wandb.watch(self.model, self.criterion, log="all")
                with BatchMemoryManager(data_loader=self.train_dl, max_physical_batch_size = self.args.max_physical_batch_size, optimizer=self.optimizer) as new_data_loader: 
                    for batch_idx, (ecgs, ag, labels) in enumerate(new_data_loader,0):
                        ecgs = ecgs.to(self.device) # ECGs
                        ag = ag.to(self.device) # age and gender
                        labels = labels.to(self.device) # diagnoses in SNOMED CT codes  
                        self.args.logger.info('input batch size () '.format(len(labels)))

                        with torch.set_grad_enabled(True):                    

                            logits = self.model(ecgs, ag) 
                            loss = self.criterion(logits, labels)
                            logits_prob = self.sigmoid(logits)      
                            loss_tmp = loss.item() * ecgs.size(0)
                            labels_all = torch.cat((labels_all, labels), 0)
                            logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)                    
                            
                            train_loss += loss_tmp
                            
                            self.optimizer.zero_grad()

                            loss.backward()

                            self.optimizer.step()

                            # Printing training information
                            if step % int(20000/config.batch_size) == 0:
                           
                                batch_loss += loss_tmp
                                batch_count += ecgs.size(0)
                                batch_loss = batch_loss / batch_count
                            
                            
                                self.args.logger.info('epoch {:^3} [{}/{}] train loss: {:>5.4f}'.format(
                                    epoch, 
                                    batch_idx * len(ecgs), 
                                    len(self.train_dl.dataset), 
                                    batch_loss
                                    
                                )) 
                                
                                batch_loss = 0.0
                                batch_count = 0
                                
                                train_loss = train_loss / len(self.train_dl.dataset)            
                                train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc, train_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
                                self.args.logger.info('epoch {:^4}/{:^4} train loss: {:<6.2f}  train micro auroc: {:<6.2f}  train challenge metric: {:<6.2f}'.format( 
                                                    epoch, 
                                                    self.args.epochs, 
                                                    train_loss, 
                                                    train_micro_auroc,
                                                    train_challenge_metric))
                            step += 1
                        
                train_loss = train_loss / len(self.train_dl.dataset)            
                train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc, train_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
                self.args.logger.info('epoch {:^4}/{:^4} train loss: {:<6.2f}  train micro auroc: {:<6.2f}  train challenge metric: {:<6.2f}'.format( 
                    epoch, 
                    self.args.epochs, 
                    train_loss, 
                    train_micro_auroc,
                    train_challenge_metric))
            
                wandb.log({"train_micro_auroc": train_micro_auroc,\
                                        "train_macro_auroc": train_macro_auroc,\
                                        "loss": loss,\
                                        "epochs": epoch})


                
                
            
            