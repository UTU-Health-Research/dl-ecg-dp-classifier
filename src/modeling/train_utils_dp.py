import os, sys
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .optimizer import NoamOpt
from .models.ctn_dp import CTN
from ..dataloader.dataset import ECGDataset, get_transforms
from .metrics import cal_multilabel_metrics, roc_curves
import pickle
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

class Training(object):
    def __init__(self, args):
        self.args = args
  
    def setup(self):
        '''Initializing the device conditions, datasets, dataloaders, 
        model, loss, criterion and optimizer'''
        
        # Create epsilon-specific subdirectory
        self.eps_dir = os.path.join(self.args.model_save_dir, f"eps_{self.args.epsilon}_epo_{self.args.epochs}_seed_{self.args.seed}")
        os.makedirs(self.eps_dir, exist_ok=True)
        
        # Create ROC curves directory inside the epsilon subdirectory
        self.eps_roc_dir = os.path.join(self.eps_dir, "ROC_curves")
        os.makedirs(self.eps_roc_dir, exist_ok=True)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(1)
            self.device = torch.device("cuda:1")
            self.device_count = 1
            self.args.logger.info('using gpu:1')
            
            assert self.args.batch_size % self.device_count == 0, \
                "batch size should be divisible by device count"
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            self.args.logger.info('using cpu')

        # Load the datasets       
        training_set = ECGDataset(self.args.train_path, get_transforms('train'))
        self.train_dl = DataLoader(training_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=True,
                                   num_workers=self.args.num_workers,
                                   pin_memory=(True if self.device == 'cuda' else False),
                                   drop_last=True)
        channels = training_set.channels

        if self.args.val_path is not None:
            validation_set = ECGDataset(self.args.val_path, get_transforms('val'))
            self.validation_files = validation_set.data
            self.val_dl = DataLoader(validation_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=self.args.num_workers,
                                    pin_memory=(True if self.device == 'cuda' else False),
                                    drop_last=True)

        self.model = CTN(in_channel=channels, out_channel=len(self.args.labels))

        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model) 
        
        # Use standard Adam optimizer for compatibility with Opacus
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-9)

        self.model.to(self.device)
        
        # Apply differential privacy if enabled
        if self.args.dp:
            self.privacy_engine = PrivacyEngine(
                accountant="rdp",
                secure_mode=False,
            )
            
            # Calculate noise multiplier using the utility function
            sample_rate = self.args.batch_size / len(self.train_dl.dataset)
            noise_multiplier = get_noise_multiplier(
                target_epsilon=self.args.epsilon,
                target_delta=self.args.delta,
                sample_rate=sample_rate,
                epochs=self.args.epochs,
                accountant="rdp",
            )
            
            # Create a list of max_grad_norm values for each parameter
            param_groups = len(list(self.model.parameters()))
            max_grad_norms = [self.args.max_grad_norm] * param_groups

            self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norms,
                clipping="per_layer"
            )

            # self.privacy_engine.attach(self.model, self.optimizer, self.train_dl)
        
    def train(self):
        '''PyTorch training loop'''
        
        self.args.logger.info('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
              (type(self.model).__name__, 
               type(self.optimizer).__name__,
               self.optimizer.param_groups[0]['lr'] if hasattr(self.optimizer, 'param_groups') else 0,
               self.args.epochs, 
               self.device))
        
        history = {}
        history['train_loss'] = []
        history['train_micro_auroc'] = []
        history['train_micro_avg_prec'] = []  
        history['train_macro_auroc'] = []
        history['train_macro_avg_prec'] = [] 
        
        # Add epsilon and delta to history
        history['epsilon'] = self.args.epsilon
        history['delta'] = self.args.delta

        if self.args.val_path is not None:
            history['val_csv'] = self.args.val_path
            history['val_loss'] = []
            history['val_micro_auroc'] = []
            history['val_micro_avg_prec'] = []
            history['val_macro_auroc'] = []
            history['val_macro_avg_prec'] = []

        history['labels'] = self.args.labels
        history['epochs'] = self.args.epochs
        history['batch_size'] = self.args.batch_size
        history['lr'] = self.args.lr
        history['optimizer_type'] = type(self.optimizer).__name__  # Save type instead of object
        history['criterion'] = "F.binary_cross_entropy_with_logits()"
        history['train_csv'] = self.args.train_path
        
        start_time_sec = time.time()
        
        for epoch in range(1, self.args.epochs+1):
            self.model.train()            
            train_loss = 0.0
            labels_all = torch.tensor((), device=self.device)
            logits_prob_all = torch.tensor((), device=self.device)
            
            batch_loss = 0.0
            batch_count = 0
            step = 0
            
            for batch_idx, (ecg, labels) in enumerate(self.train_dl):
                ecg = ecg.float().to(self.device)
                labels = labels.float().to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(ecg)
                logits_prob = logits.sigmoid().data
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss_tmp = loss.item()

                labels_all = torch.cat((labels_all, labels), 0)
                logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)                    
                
                train_loss += loss_tmp
                loss.backward()
                self.optimizer.step()

                if step % 100 == 0:
                    batch_loss += loss_tmp
                    batch_count += ecg.size(0)
                    batch_loss = batch_loss / batch_count
                    self.args.logger.info('epoch {:^3} [{}/{}] train loss: {:>5.4f}'.format(
                        epoch, 
                        batch_idx * len(ecg), 
                        len(self.train_dl.dataset), 
                        batch_loss
                    ))

                    batch_loss = 0.0
                    batch_count = 0
                step += 1

            train_loss = train_loss / len(self.train_dl.dataset)            
            train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)

            history['train_loss'].append(train_loss)
            history['train_micro_auroc'].append(train_micro_auroc)
            history['train_micro_avg_prec'].append(train_micro_avg_prec)
            history['train_macro_auroc'].append(train_macro_auroc)
            history['train_macro_avg_prec'].append(train_macro_avg_prec)

            self.args.logger.info('epoch {:^4}/{:^4} train loss: {:<6.2f}  train macro auroc: {:<6.2f} '.format( 
                epoch, 
                self.args.epochs, 
                train_loss, 
                train_macro_auroc))

            if self.args.val_path is not None:
                self.model.eval()
                val_loss = 0.0  
                labels_all = torch.tensor((), device=self.device)
                logits_prob_all = torch.tensor((), device=self.device)  
                
                for ecg, labels in self.val_dl:
                    ecg = ecg.float().to(self.device)
                    labels = labels.float().to(self.device)
                    
                    with torch.no_grad():  
                        logits = self.model(ecg)
                        loss = F.binary_cross_entropy_with_logits(logits, labels)
                        logits_prob = logits.sigmoid().data
                        val_loss += loss.item() * ecg.size(0)  

                        labels_all = torch.cat((labels_all, labels), 0)
                        logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

                val_loss = val_loss / len(self.val_dl.dataset)
                val_macro_avg_prec, val_micro_avg_prec, val_macro_auroc, val_micro_auroc = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
            
                history['val_loss'].append(val_loss)
                history['val_micro_auroc'].append(val_micro_auroc)
                history['val_micro_avg_prec'].append(val_micro_avg_prec)         
                history['val_macro_auroc'].append(val_macro_auroc)  
                history['val_macro_avg_prec'].append(val_macro_avg_prec)
        
                self.args.logger.info('                val loss:  {:<6.2f}   val macro auroc: {:<6.2f}'.format(
                    val_loss,
                    val_macro_auroc))

            # Create ROC Curves only at the last epoch (modified)
            if epoch == self.args.epochs:
                roc_curves(labels_all, logits_prob_all, self.args.labels, epoch, self.eps_roc_dir, dp_suffix=f'_eps{self.args.epsilon}_dp')

            if epoch == self.args.epochs:
                self.args.logger.info('Saving the model, training history and validation logits...')
                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                model_savepath = os.path.join(self.eps_dir,
                                              f"{self.args.yaml_file_name}_dp.pth")
                torch.save(model_state_dict, model_savepath)
                
                history_savepath = os.path.join(self.eps_dir,
                                                f"{self.args.yaml_file_name}_dp_train_history.pickle")
                with open(history_savepath, mode='wb') as file:
                    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
                if self.args.val_path is not None:
                    self.args.logger.info('- Validation logits and labels saved')
                    logits_csv_path = os.path.join(self.eps_dir,
                                               f"{self.args.yaml_file_name}_dp_val_logits.csv") 
                    labels_all_csv_path = os.path.join(self.eps_dir,
                                                f"{self.args.yaml_file_name}_dp_val_labels.csv") 
                    filenames = [os.path.basename(file) for file in self.validation_files]
                else:
                    self.args.logger.info('- Training logits and actual labels saved (no validation set available)')
                    logits_csv_path = os.path.join(self.eps_dir,
                                               f"{self.args.yaml_file_name}_dp_train_logits.csv") 
                    labels_all_csv_path = os.path.join(self.eps_dir,
                                                f"{self.args.yaml_file_name}_dp_train_labels.csv") 
                    filenames = None
                
                labels_numpy = labels_all.cpu().detach().numpy().astype(np.float32)
                labels_df = pd.DataFrame(labels_numpy, columns=self.args.labels, index=filenames)
                labels_df.to_csv(labels_all_csv_path, sep=',')

                logits_numpy = logits_prob_all.cpu().detach().numpy().astype(np.float32)
                logits_df = pd.DataFrame(logits_numpy, columns=self.args.labels, index=filenames)
                logits_df.to_csv(logits_csv_path, sep=',')

            del logits_prob_all
            del labels_all
            torch.cuda.empty_cache()

        # Report privacy budget if differential privacy is enabled
        if self.args.dp:
            # epsilon = self.privacy_engine.get_epsilon(delta=self.args.delta)
            epsilon = self.privacy_engine.get_epsilon(delta=self.args.delta)
            self.args.logger.info(
                f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for {self.args.epochs} epochs"
            )
            # self.args.logger.info(f"Final privacy budget: epsilon={epsilon}, best_alpha={best_alpha}")

        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / self.args.epochs
        self.args.logger.info('Time total:     %5.2f sec' % (total_time_sec))
        self.args.logger.info('Time per epoch: %5.2f sec' % (time_per_epoch_sec))
        
        # Run evaluation on test set if test_path is provided
        if hasattr(self.args, 'test_path') and self.args.test_path is not None:
            self.evaluate(self.args.test_path)
        
    def evaluate(self, test_path):
        '''Evaluate the model on test data
        
        :param test_path: Path to test CSV file
        :type test_path: str
        '''
        self.args.logger.info(f'Evaluating model on test data: {test_path}')
        
        # Find test files based on the test csv (for naming saved predictions)
        # The paths for these files are in the 'path' column
        filenames = pd.read_csv(test_path, usecols=['path']).values.tolist()
        filenames = [f for file in filenames for f in file]
        
        # Load the test data
        testing_set = ECGDataset(test_path, get_transforms('test'))
        test_dl = DataLoader(testing_set,
                             batch_size=1,
                             shuffle=False,
                             num_workers=self.args.num_workers,
                             pin_memory=(True if self.device == 'cuda' else False),
                             drop_last=True)
        
        # Create history dictionary to store evaluation metrics
        history = {}
        history['test_micro_avg_prec'] = 0.0
        history['test_micro_auroc'] = 0.0
        history['test_macro_avg_prec'] = 0.0
        history['test_macro_auroc'] = 0.0
        history['labels'] = self.args.labels
        history['test_csv'] = test_path
        history['threshold'] = self.args.threshold
        # Add epsilon and delta to history
        history['epsilon'] = self.args.epsilon
        history['delta'] = self.args.delta
        
        start_time_sec = time.time()
        
        # Set model to evaluation mode
        self.model.eval()
        labels_all = torch.tensor((), device=self.device)
        logits_prob_all = torch.tensor((), device=self.device)
        
        for i, (ecg, labels) in enumerate(test_dl):
            ecg = ecg.float().to(self.device)
            labels = labels.float().to(self.device)
            
            with torch.no_grad():  
                logits = self.model(ecg)
                logits_prob = logits.sigmoid().data                        
                labels_all = torch.cat((labels_all, labels), 0)
                logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)
            
            if i % 1000 == 0:
                self.args.logger.info('{:<4}/{:>4} test samples processed'.format(i+1, len(test_dl)))
        
        # Calculate metrics
        test_macro_avg_prec, test_micro_avg_prec, test_macro_auroc, test_micro_auroc = cal_multilabel_metrics(
            labels_all, logits_prob_all, self.args.labels, self.args.threshold
        )
        
        self.args.logger.info('Test metrics:')
        self.args.logger.info('macro avg prec: {:<6.2f} micro avg prec: {:<6.2f} macro auroc: {:<6.2f} micro auroc: {:<6.2f}'.format(
            test_macro_avg_prec,
            test_micro_avg_prec,
            test_macro_auroc,
            test_micro_auroc
        ))
        
        # Draw ROC curve for predictions
        roc_curves(labels_all, logits_prob_all, self.args.labels, save_path=self.eps_dir, dp_suffix=f'_dp')
        
        # Add metrics to history
        history['test_micro_auroc'] = test_micro_auroc
        history['test_micro_avg_prec'] = test_micro_avg_prec
        history['test_macro_auroc'] = test_macro_auroc
        history['test_macro_avg_prec'] = test_macro_avg_prec
        
        # Save history
        history_savepath = os.path.join(self.eps_dir,
                                      f"{self.args.yaml_file_name}_dp_test_history.pickle")
        with open(history_savepath, mode='wb') as file:
            pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save logits and labels
        filenames = [os.path.basename(file) for file in filenames]
        
        logits_csv_path = os.path.join(self.eps_dir,
                                     f"{self.args.yaml_file_name}_dp_test_logits.csv") 
        logits_numpy = logits_prob_all.cpu().detach().numpy().astype(np.float32)
        logits_df = pd.DataFrame(logits_numpy, columns=self.args.labels, index=filenames)
        logits_df.to_csv(logits_csv_path, sep=',')
        
        labels_csv_path = os.path.join(self.eps_dir,
                                      f"{self.args.yaml_file_name}_dp_test_labels.csv") 
        labels_numpy = labels_all.cpu().detach().numpy().astype(np.float32)
        labels_df = pd.DataFrame(labels_numpy, columns=self.args.labels, index=filenames)
        labels_df.to_csv(labels_csv_path, sep=',')
        
        torch.cuda.empty_cache()
        
        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        self.args.logger.info('Test evaluation time: %5.2f sec' % (total_time_sec))