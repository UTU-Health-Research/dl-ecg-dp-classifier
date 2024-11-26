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
    
    def __init__(self, args):
        self.args = args
        
        
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


    def train(self):
        ''' PyTorch training loop
        '''
       
        if self.args.batch_size < 400: 
            self.args.max_physical_batch_size = self.args.batch_size
            print('max_physical_batch_size is : ', self.max_physical_batch_size)

        training_set = ECGDataset(self.args.train_path, get_transforms('train'), amount = None)
        channels = training_set.channels
        self.train_dl = DataLoader(training_set,
                                batch_size= self.args.batch_size ,
                                shuffle=True,
                                num_workers=self.args.num_workers,
                                pin_memory=(True if self.device == 'cuda' else False),
                                drop_last=False)

        # if self.args.val_path is not None:
        #     validation_set = ECGDataset(self.args.val_path, get_transforms('val'), amount = 100)
        #     self.validation_files = validation_set.data
        #     self.val_dl = DataLoader(validation_set,
        #                             batch_size=1,
        #                             shuffle=False,
        #                             num_workers=self.args.num_workers,
        #                             pin_memory=(True if self.device == 'cuda' else False),
        #                             drop_last=True)

        # filenames = pd.read_csv(self.args.test_path, usecols=['path']).values.tolist()
        # self.filenames = [f for file in filenames for f in file]

        # # Load the test data
        # testing_set = ECGDataset(self.args.test_path, 
        #                         get_transforms('test'),amount=None)
        # channels = testing_set.channels
        # self.test_dl = DataLoader(testing_set,
        #                         batch_size=1,
        #                         shuffle=False,
        #                         pin_memory=(True if self.device == 'cuda' else False),
        #                         drop_last=True)

        self.args.logger.info('total training samples is {}'.format(len(training_set)))
        # self.args.logger.info('total validation samples is {}'.format(len(validation_set)))
        # self.args.logger.info('total testing samples is {}'.format(len(testing_set)))

        self.model = resnet18_v2(in_channel=channels, 
                            out_channel=len(self.args.labels))

        # Load model if necessary
        if hasattr(self.args, 'load_model_path'):
            self.model.load_state_dict(torch.load(self.args.load_model_path))
            self.args.logger.info('Loaded the model from: {}'.format(self.args.load_model_path))
        else:
            self.args.logger.info('Training a new model from the beginning.')
        
        # If more than 1 CUDA device used, use data parallelism
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model) 
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)

        self.delta =  1/len(self.train_dl)

        if (self.args.with_privacy):
            self.args.logger.info('Training with DP-SGD algorithm')
            self.args.logger.info('Training DP with accountant{}'.format(self.args.accountant))
            self.privacy_engine = PrivacyEngine(accountant=self.args.accountant)
            # Attach Privacy Engine

            self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                target_epsilon= self.args.epsilon,
                target_delta= self.delta,
                epochs = self.args.epochs,
                max_grad_norm = self.args.max_grad_norm
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
    
    
        for epoch in range(1, self.args.epochs+1):
            # --- TRAIN ON TRAINING SET ------------------------------------------
            self.model.train()            
            train_loss = 0.0
            labels_all = torch.tensor((), device=self.device) # , device=torch.device('cuda:0')
            logits_prob_all = torch.tensor((), device=self.device)
            
            batch_loss = 0.0
            batch_count = 0
            step = 0

            with BatchMemoryManager(data_loader=self.train_dl, max_physical_batch_size = self.args.max_physical_batch_size, optimizer=self.optimizer) as new_data_loader: 
                for batch_idx, (ecgs, ag, labels) in enumerate(new_data_loader,0):
                    ecgs = ecgs.to(self.device) # ECGs
                    ag = ag.to(self.device) # age and gender
                    labels = labels.to(self.device) # diagnoses in SNOMED CT codes  
                
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
                        valid_gradients = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]

                        original_gradients =  sum(valid_gradients)/ len(valid_gradients)
                        

                        self.optimizer.step()

                        # valid_gradients = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]

                        # gradients_af_noise = sum(valid_gradients)/ len(valid_gradients)
                    
                        # Printing training information
                        if step % int(20000/self.args.batch_size) == 0:
                            # print('gradient before noise', original_gradients)
                            # print('gradient after clipping', self.optimizer.abs_avg_grad_ac)
                            # print('gradient after noise', gradients_af_noise)
                            

                            batch_loss += loss_tmp
                            batch_count += ecgs.size(0)
                            batch_loss = batch_loss / batch_count
                        
                        
                            self.args.logger.info('epoch {:^3} [{}/{}] train loss: {:>5.4f}'.format(
                                epoch, 
                                batch_idx * len(ecgs), 
                                len(self.train_dl.dataset), 
                                batch_loss
                                
                            )) 
                            
                            # batch_loss = 0.0
                            # batch_count = 0
                            
                            # train_loss = train_loss / len(self.train_dl.dataset)            
                            # train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc, train_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
                            
                            # if (self.args.with_privacy):
                            #     epsilon = self.privacy_engine.get_epsilon(self.delta)
                            #     wandb.log({"micro_avg": train_micro_auroc,\
                            #             "macro_avg": train_macro_auroc,\
                            #             "loss": loss,\
                            #             "epsilon":epsilon,
                            #             "original_gradients": original_gradients,\
                            #             "gradients_af_noise": gradients_af_noise,\
                            #             "gradients_af_clipping": self.optimizer.abs_avg_grad_ac,\
                            #             }) ## logging the loss curve
                            # else:
                            #     wandb.log({"micro_avg": train_micro_auroc,\
                            #             "macro_avg": train_macro_auroc,\
                            #             "loss": loss})
                            
                        step += 1
                    
            train_loss = train_loss / len(self.train_dl.dataset)            
            train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc, train_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
            self.args.logger.info('epoch {:^4}/{:^4} train loss: {:<6.2f}  train micro auroc: {:<6.2f}  train challenge metric: {:<6.2f}'.format( 
                epoch, 
                self.args.epochs, 
                train_loss, 
                train_micro_auroc,
                train_challenge_metric))
            # print('epoch {:^4}/{:^4} train loss: {:<6.2f}  train micro auroc: {:<6.2f}  train challenge metric: {:<6.2f}'.format( 
            #     epoch, 
            #     self.args.epochs, 
            #     train_loss, 
            #     train_micro_auroc,
            #     train_challenge_metric))


            # Add information for training history
            # history['train_loss'].append(train_loss)
            # history['train_micro_auroc'].append(train_micro_auroc)
            # history['train_micro_avg_prec'].append(train_micro_avg_prec)
            # history['train_macro_auroc'].append(train_macro_auroc)
            # history['train_macro_avg_prec'].append(train_macro_avg_prec)
            # history['train_challenge_metric'].append(train_challenge_metric)
            
            
            #--- EVALUATE ON VALIDATION SET ------------------------------------- 
        #     if self.args.val_path is not None:
        #         self.model.eval()
        #         val_loss = 0.0  
        #         labels_all = torch.tensor((), device=self.device)
        #         logits_prob_all = torch.tensor((), device=self.device)  
            
        #         for ecgs, ag, labels in self.val_dl:
        #             ecgs = ecgs.to(self.device) # ECGs
        #             ag = ag.to(self.device) # age and gender
        #             labels = labels.to(self.device) # diagnoses in SNOMED CT codes 
                    
        #             with torch.set_grad_enabled(False):  
                        
        #                 logits = self.model(ecgs, ag)
        #                 loss = self.criterion(logits, labels)
        #                 logits_prob = self.sigmoid(logits)
        #                 val_loss += loss.item() * ecgs.size(0)                                 
        #                 labels_all = torch.cat((labels_all, labels), 0)
        #                 logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

        #         val_loss = val_loss / len(self.val_dl.dataset)
        #         val_macro_avg_prec, val_micro_avg_prec, val_macro_auroc, val_micro_auroc, val_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
            
        #     history['val_loss'].append(val_loss)
        #     history['val_micro_auroc'].append(val_micro_auroc)
        #     history['val_micro_avg_prec'].append(val_micro_avg_prec)         
        #     history['val_macro_auroc'].append(val_macro_auroc)  
        #     history['val_macro_avg_prec'].append(val_macro_avg_prec)
        #     history['val_challenge_metric'].append(val_challenge_metric)
        
        # self.args.logger.info('Test Prediction')
        # # --- EVALUATE ON TESTING SET -------------------------------------     
        # if self.args.test_path is not None:
        #     self.model.eval()
        #     labels_all = torch.tensor((), device=self.device)
        #     logits_prob_all = torch.tensor((), device=self.device)  
            
        #     for i, (ecgs, ag, labels) in enumerate(self.test_dl):
        #         ecgs = ecgs.to(self.device) # ECGs
        #         ag = ag.to(self.device) # age and gender
        #         labels = labels.to(self.device) # diagnoses in SMONED CT codes 

        #         with torch.set_grad_enabled(False):  
                    
        #             logits = self.model(ecgs, ag)
        #             logits_prob = self.sigmoid(logits)
        #             labels_all = torch.cat((labels_all, labels), 0)
        #             logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

            
        #         # ------ One-hot-encode predicted label -----------
        #         # Define an empty label for predictions
        #         pred_label = np.zeros(len(self.args.labels))

        #         # Find the maximum values within the probabilities
        #         _, likeliest_dx = torch.max(logits_prob, 1)

        #         # Predicted probabilities from tensor to numpy
        #         likeliest_dx = likeliest_dx.cpu().detach().numpy()

        #         # First, add the most likeliest diagnosis to the predicted label
        #         pred_label[likeliest_dx] = 1

        #         # Then, add all the others that are above the decision threshold
        #         other_dx = logits_prob.cpu().detach().numpy() >= self.args.threshold
        #         pred_label = pred_label + other_dx
        #         pred_label[pred_label > 1.1] = 1
        #         pred_label = np.squeeze(pred_label)

        #         # --------------------------------------------------
                
        #         # Save also probabilities but return them first in numpy
        #         scores = logits_prob.cpu().detach().numpy()
        #         scores = np.squeeze(scores)
                
        #         # Save the prediction
        #         #self.save_predictions(self.filenames[i], pred_label, scores, self.args.pred_save_dir)

        #         if i % 1000 == 0:
        #             self.args.logger.info('{:<4}/{:>4} predictions made'.format(i+1, len(self.test_dl)))

        #     # Predicting metrics
        #     test_macro_avg_prec, test_micro_avg_prec, test_macro_auroc, test_micro_auroc, test_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
        #     history['test_micro_auroc'] = test_micro_auroc
        #     history['test_macro_auroc'] = test_macro_auroc

        # # model_path = wandb.run.dir+"/model.onnx"
        # # torch.onnx.export(self.model, (ecgs, ag),model_path)
        # # wandb.save(model_path)   

        # self.args.logger.info('Saving all the configurations')
        # history_path = wandb.run.dir+'/model_history.pkl'
        # with open(history_path, 'wb') as f:
        #     pickle.dump(history, f)
        
        # wandb.save(history_path)        



                    
