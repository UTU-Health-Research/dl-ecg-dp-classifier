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

class Training(object):
    
    def __init__(self, args):
        self.args = args
        

    def setup(self):
        '''Initializing the device conditions, datasets, dataloaders, 
        model, loss, criterion and optimizer
        '''
        start_time_sec = time.time()

        # Consider the GPU or CPU condition
        if torch.cuda.is_available():
            gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "1")  # Get the GPU index from CUDA_VISIBLE_DEVICES
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
            self.device = torch.device("cuda")
            self.device_count = self.args.device_count
            self.args.logger.info('using gpu device {}'.format(gpu_index))
            assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            self.args.logger.info('using {} cpu'.format(self.device_count))

        # Load the datasets       
        training_set = ECGDataset(self.args.train_path, get_transforms('train'), amount = None)
        channels = training_set.channels
        self.train_dl = DataLoader(training_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=True,
                                   num_workers=self.args.num_workers,
                                   pin_memory=(True if self.device == 'cuda' else False),
                                   drop_last=False)

        if self.args.val_path is not None:
            validation_set = ECGDataset(self.args.val_path, get_transforms('val'), amount = None)
            self.validation_files = validation_set.data
            self.val_dl = DataLoader(validation_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=self.args.num_workers,
                                    pin_memory=(True if self.device == 'cuda' else False),
                                    drop_last=True)

        self.args.logger.info('total training samples is {}'.format(len(training_set)))
        self.args.logger.info('total validation samples is {}'.format(len(validation_set)))

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
            privacy_engine = PrivacyEngine(accountant=self.args.accountant)
            self.model, self.optimizer, self.data_loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                target_epsilon=self.args.epsilon,
                target_delta= self.delta,
                epochs = self.args.epochs,
                max_grad_norm = self.args.max_grad_norm
                )
       
            print(f"Noise: {self.optimizer.noise_multiplier}")
            
        self.criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)

        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec
        self.total_setup_time_sec = total_time_sec
        

    def train(self):
        ''' PyTorch training loop
        '''
        with wandb.init(project="ECG", name=self.args.run_name, config = self.args):
            self.args.logger.info('train() called: model={}, opt={}(lr={}), epochs={}, device={}'.format(
                    type(self.model).__name__, 
                    type(self.optimizer).__name__,
                    self.optimizer.param_groups[0]['lr'], 
                    self.args.epochs, 
                    self.device))
            
            # Add all wanted history information
            history = {}
            history['train_csv'] = self.args.train_path
            history['train_loss'] = []
            history['train_micro_auroc'] = []
            history['train_micro_avg_prec'] = []  
            history['train_macro_auroc'] = []
            history['train_macro_avg_prec'] = [] 
            history['train_challenge_metric'] = []
            history['total_training_time_sec'] =[]
            history['epsilon'] = -1
            history['total_setup_time_sec'] = self.total_setup_time_sec
            history['batch_size'] = self.args.batch_size
            history['max_grad_norm'] = self.args.max_grad_norm
            history['noise_multiplier'] = 0
            history['labels'] = self.args.labels
            history['epochs'] = self.args.epochs
            history['lr'] = self.args.lr
            history['criterion'] = self.criterion
            history['clipping'] = self.args.clipping
            history['avg_gradients'] = []
            # history['min_gradients'] = []
            # history['max_gradients'] = []
            if self.args.val_path is not None:
                history['val_csv'] = self.args.val_path
                history['val_loss'] = []
                history['val_micro_auroc'] = []
                history['val_micro_avg_prec'] = []
                history['val_macro_auroc'] = []
                history['val_macro_avg_prec'] = []
                history['val_challenge_metric'] = []
            
            if self.args.clipping:
                self.args.logger.info('Applying Clipping')
            
            start_time_sec = time.time()
            wandb.watch(self.model, self.criterion, log="all", log_freq=10)

            for epoch in range(1, self.args.epochs+1):
                # --- TRAIN ON TRAINING SET ------------------------------------------
                self.model.train()            
                train_loss = 0.0
                labels_all = torch.tensor((), device=self.device) # , device=torch.device('cuda:0')
                logits_prob_all = torch.tensor((), device=self.device)
                
                batch_loss = 0.0
                batch_count = 0
                step = 0
                
                for batch_idx, (ecgs, ag, labels) in enumerate(self.train_dl):
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
                        
                    
                        self.optimizer.step()
                        
                        if self.args.clipping:
                                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)
                            

                        # Printing training information
                        if step % 10 == 0:
                            batch_loss += loss_tmp
                            batch_count += ecgs.size(0)
                            batch_loss = batch_loss / batch_count
                            

                                # grads = [
                                #         param.grad.clone().detach().flatten()
                                #         for param in self.model.parameters()
                                #         if param.grad is not None
                                #     ]
                                # norm = torch.cat(grads).norm()
                                # avg_norm_gradients = torch.mean(torch.abs(torch.cat(grads)/norm))
                                # min_norm_gradients = torch.min(torch.abs(torch.cat(grads)/norm))
                                # max_norm_gradients = torch.max(torch.abs(torch.cat(grads)/norm))
                                
                                # clipping the gradients
                                
                            grads = [
                                        param.grad.clone().detach().flatten()
                                        for param in self.model.parameters()
                                        if param.grad is not None
                                ]
                                
                            avg_gradients = torch.mean(torch.abs(torch.cat(grads)))
                            min_gradients = torch.min(torch.abs(torch.cat(grads)))
                            max_gradients = torch.max(torch.abs(torch.cat(grads)))

                            self.args.logger.info('epoch {:^3} [{}/{}] train loss: {:>5.4f}'.format(
                                epoch, 
                                batch_idx * len(ecgs), 
                                len(self.train_dl.dataset), 
                                batch_loss
                                #avg_gradients,min_gradients,max_gradients
                            ))
                            self.args.logger.info('avg_gradients= {},min_gradients = {},max_gradients = {}'.format(avg_gradients,min_gradients,max_gradients))
                            
                            history['avg_gradients'].append(avg_gradients)
                            batch_loss = 0.0
                            batch_count = 0
                            wandb.log({"max_grad": max_gradients, "loss": loss}, step=step)
                        
                        step += 1
                
                
                
                train_loss = train_loss / len(self.train_dl.dataset)            
                train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc, train_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)

                self.args.logger.info('epoch {:^4}/{:^4} train loss: {:<6.2f}  train micro auroc: {:<6.2f}  train challenge metric: {:<6.2f}'.format( 
                    epoch, 
                    self.args.epochs, 
                    train_loss, 
                    train_micro_auroc,
                    train_challenge_metric))
                print('epoch {:^4}/{:^4} train loss: {:<6.2f}  train micro auroc: {:<6.2f}  train challenge metric: {:<6.2f}'.format( 
                    epoch, 
                    self.args.epochs, 
                    train_loss, 
                    train_micro_auroc,
                    train_challenge_metric))

                # Add information for training history
                history['train_loss'].append(train_loss)
                history['train_micro_auroc'].append(train_micro_auroc)
                history['train_micro_avg_prec'].append(train_micro_avg_prec)
                history['train_macro_auroc'].append(train_macro_auroc)
                history['train_macro_avg_prec'].append(train_macro_avg_prec)
                history['train_challenge_metric'].append(train_challenge_metric)
                
                #--- EVALUATE ON VALIDATION SET ------------------------------------- 
                if self.args.val_path is not None:
                    self.model.eval()
                    val_loss = 0.0  
                    labels_all = torch.tensor((), device=self.device)
                    logits_prob_all = torch.tensor((), device=self.device)  
                
                    for ecgs, ag, labels in self.val_dl:
                        ecgs = ecgs.to(self.device) # ECGs
                        ag = ag.to(self.device) # age and gender
                        labels = labels.to(self.device) # diagnoses in SNOMED CT codes 
                        
                        with torch.set_grad_enabled(False):  
                            
                            logits = self.model(ecgs, ag)
                            loss = self.criterion(logits, labels)
                            logits_prob = self.sigmoid(logits)
                            val_loss += loss.item() * ecgs.size(0)                                 
                            labels_all = torch.cat((labels_all, labels), 0)
                            logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

                    val_loss = val_loss / len(self.val_dl.dataset)
                    val_macro_avg_prec, val_micro_avg_prec, val_macro_auroc, val_micro_auroc, val_challenge_metric = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
                
                    self.args.logger.info('                val loss:  {:<6.2f}   val micro auroc: {:<6.2f}    val challenge metric:  {:<6.2f}'.format(
                        val_loss,
                        val_micro_auroc,
                        val_challenge_metric))
                    
                    history['val_loss'].append(val_loss)
                    history['val_micro_auroc'].append(val_micro_auroc)
                    history['val_micro_avg_prec'].append(val_micro_avg_prec)         
                    history['val_macro_auroc'].append(val_macro_auroc)  
                    history['val_macro_avg_prec'].append(val_macro_avg_prec)
                    history['val_challenge_metric'].append(val_challenge_metric)

                # --------------------------------------------------------------------
                add_DP_path = '_without_DP'
                if (self.args.with_privacy):
                    add_DP_path = f'_with_DP_epsilon_{self.args.epsilon}'
                    history['epsilon'] = self.args.epsilon
                    history['noise_multiplier'] = self.optimizer.noise_multiplier
                
                    
                # # Save trained model (.pth), history (.pickle) and validation logits (.csv) after the last epoch
                # if epoch == self.args.epochs:

                #     roc_curves_dic = roc_curves(labels_all, logits_prob_all, self.args.labels, epoch, None ,None)
                #     history['roc_curves'] = roc_curves_dic
                    
                #     self.args.logger.info('Saving the model, training history and validation logits...')
                        
                #     # Whether or not you use data parallelism, save the state dictionary this way
                #     # to have the flexibility to load the model any way you want to any device you want
                #     #model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                        
                    
                #     # -- Save model
                #     model_savepath = os.path.join(self.args.model_save_dir,
                #                                 'model' + add_DP_path +'_clipping_value_'+str(self.args.max_grad_norm)+ '.pth')
                #     torch.save(self.model, model_savepath)
                    
                #     # -- Save history
                #     history_savepath = os.path.join(self.args.history_save_dir,
                #                                     'model' + add_DP_path +'_clipping_value_'+str(self.args.max_grad_norm)+'.pickle')
                #     with open(history_savepath, mode='wb') as file:
                #         pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
                        
                    # -- Save the logits from validation if used, either save the logits from the training phase
                    # if self.args.val_path is not None:
                    #     self.args.logger.info('- Validation logits saved')
                    #     logits_csv_path = os.path.join(self.args.model_save_dir,
                    #                                self.args.yaml_file_name + '_val_logits.csv') 
                    #     # Cleanup filenames to use as indexes
                    #     cleanup_filenames = [os.path.basename(file) for file in self.validation_files]

                    # else:
                    #     self.args.logger.info('- Training logits and actual labels saved (no validation set available)')
                    #     logits_csv_path = os.path.join(self.args.model_save_dir,
                    #                                self.args.yaml_file_name + '_train_logits.csv') 
                    #     cleanup_filenames = None
                        
                    #     # If only training used and the logits saved from there,
                    #     # save also actual labels as the DataLoader is shuffled
                    #     labels_all_csv_path = os.path.join(self.args.model_save_dir,
                    #                                self.args.yaml_file_name + '_actual_labels.csv') 
                    #     labels_numpy = labels_all.cpu().detach().numpy().astype(np.float32)
                    #     labels_df = pd.DataFrame(labels_numpy, columns=self.args.labels, index=cleanup_filenames)
                    #     labels_df.to_csv(labels_all_csv_path, sep=',')
                        

                    # Save the logits as a csv file where columns are the labels and 
                    # indexes are the files which have been used in the validation phase
                    # logits_numpy = logits_prob_all.cpu().detach().numpy().astype(np.float32)
                    # logits_df = pd.DataFrame(logits_numpy, columns=self.args.labels, index=cleanup_filenames)
                    # logits_df.to_csv(logits_csv_path, sep=',')

            torch.cuda.empty_cache()
            
            
            # END OF TRAINING LOOP        
            
            end_time_sec       = time.time()
            total_time_sec     = end_time_sec - start_time_sec
            time_per_epoch_sec = total_time_sec / self.args.epochs
            self.args.logger.info('Time total:     %5.2f sec' % (total_time_sec))
            self.args.logger.info('Time per epoch: %5.2f sec' % (time_per_epoch_sec))
            history['total_training_time_sec'] = history['total_training_time_sec'].append(time_per_epoch_sec)