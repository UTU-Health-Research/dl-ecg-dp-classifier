import os, sys
sys.path.append('....../src')
import time
import json
import random
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from modeling.models.seresnet18_v1 import resnet18_v1
from modeling.models.seresnet18_v2 import resnet18_v2
from modeling.models.seresnet18_v3 import resnet18_v3
from modeling.models.seresnet18_v4 import resnet18_v4
from modeling.models.seresnet18_v5 import resnet18_v5
from dataloader.dataset import ECGDataset, get_transforms
from modeling.metrics import cal_multilabel_metrics, roc_curves ,cal_classification_report
import pickle
from opacus import PrivacyEngine
import wandb
import matplotlib.pyplot as plt
import logging
from opacus.utils.batch_memory_manager import BatchMemoryManager
from pathlib import Path
import warnings
import json
import onnx
from onnx2torch import convert
from sklearn.model_selection import train_test_split
warnings.filterwarnings(action='ignore')
import logging 
logging.basicConfig(format='%(asctime)s %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S') 
logger = logging.getLogger(__name__) 
logger.setLevel(logging.DEBUG) 

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)    
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

parent_path = '/home/......../'

class Training_WANDB(object):
    
    def __init__(self, args,sweep_config):
        self.args = args
        self.sweep_config = sweep_config
        
    def setup(self):
        '''Initializing the device conditions, datasets, dataloaders, 
        model, loss, criterion and optimizer
        '''
        # Consider the GPU or CPU condition
        if torch.cuda.is_available():
            #gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "1")  # Get the GPU index from CUDA_VISIBLE_DEVICES
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_index
            self.device = torch.device(f"cuda:{self.args.gpu_index}")
            self.device_count = self.args.device_count
            # assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            self.device = torch.device("cpu")
            self.device_count = 1

        self.labels = pd.read_csv(self.args.train_path, nrows=0).columns.tolist()[4:]
        
        self.max_physical_batch_size = self.args.max_physical_batch_size
       
    def train_with_sweep(self):
        self.sweep_id = wandb.sweep(self.sweep_config, project= self.args.project_name)
        wandb.agent(self.sweep_id,self.train, count=self.args.sweep_count)

    def train(self):
        ''' PyTorch training loop
        '''
        with wandb.init(config = None) as self.run: 
            self.config = wandb.config 
            self.run.name = str(self.args.model_version)+'_'+str(self.args.source_name) +'_eps_'+str(self.args.epsilon) + '_batch_'+ str(self.config.batch_size)\
                +'_epochs_'+str(self.config.epochs) + '_seed_'+str(self.args.random_seed)
            # Check if the logical batch size (origanal batch size) is big them the memory will crash and we need to split it down into samller physical batches
            if self.config.batch_size < 400: 
                self.args.max_physical_batch_size = self.config.batch_size
                logger.info(f'max_physical_batch_size is {self.args.max_physical_batch_size}')

            # Creating DataLoader object for the train samples
            training_set = ECGDataset(get_transforms('train'),target_df=None, 
                                      path= self.args.train_path, amount = None)
            channels = training_set.channels
            def seed_worker(worker_id):
                worker_seed = self.args.random_seed

            g = torch.Generator()
            g.manual_seed(self.args.random_seed)
            
            self.train_dl = DataLoader(training_set,
                                    batch_size= self.config.batch_size,
                                    shuffle=True,
                                    num_workers=self.args.num_workers,
                                    worker_init_fn = seed_worker,
                                    generator = g,
                                    pin_memory=(True if self.device == 'cuda' else False),
                                    drop_last=False)
  
            logger.info('total training samples is {}'.format(len(training_set)))
            
            self.run.summary['train_samples_len'] = len(training_set)
           
            # defining the resnet model latest version
            if self.args.model_version in ['resnet18_v1','resnet18_v1_1_2']:
                self.model = resnet18_v1(self.args.random_seed ,in_channel=channels, out_channel=len(self.labels))
                logger.info('model version is :resnet18_v1') 

            elif self.args.model_version in ['resnet18_v2', 'resnet18_v2_1_2']:
                self.model = resnet18_v2(self.args.random_seed,in_channel=channels, out_channel=len(self.labels))
                logger.info('model version is :resnet18_v2') 

            elif self.args.model_version in ['resnet18_v3', 'resnet18_v3_1_2']:
                self.model = resnet18_v3(self.args.random_seed, in_channel=channels, out_channel=len(self.labels))
                logger.info('model version is :resnet18_v3') 

            elif self.args.model_version in ['resnet18_v4', 'resnet18_v4_1_2']:
                self.model = resnet18_v4(self.args.random_seed,in_channel=channels, out_channel=len(self.labels))
                logger.info('model version is :resnet18_v4') 
            else: 
                self.model = resnet18_v5(self.args.random_seed,in_channel=channels, out_channel=len(self.labels))
                logger.info('model version is :resnet18_v5')                

            total_params = sum(p.numel() for p in self.model.parameters())

            
            logger.info('Total model param is: {}'.format(total_params))

            self.run.summary['model_version'] = self.args.model_version
            self.run.summary['model_total_param'] = total_params

            # If more than 1 CUDA device used, use data parallelism 
            # Note here if you are using Two GPU then the way you export the model as Onnex formte has to be change
            if self.device_count > 1:
                self.model = torch.nn.DataParallel(self.model)
                logger.info('training in parralle with {} GPU'.format(self.device_count)) 
            
            # Optimizer
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr= self.config.lr,
                                        weight_decay=self.args.weight_decay)

            self.delta =  1/len(self.train_dl)

            ''' 
                    Here we define the Privacy Engine we want to add to the model
            '''
            if (self.args.with_privacy): ## we want privacy
                logger.info('Training with DP-SGD algorithm')
                logger.info('Training DP with accountant{}'.format(self.args.accountant))
                
                self.privacy_engine = PrivacyEngine(accountant=self.args.accountant)
                self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_dl,
                    target_epsilon= self.config.epsilon,
                    target_delta= self.delta,
                    epochs = self.config.epochs,
                    max_grad_norm = self.config.max_grad_norm
                    )
        
                logger.info('Noise:  {}'.format(self.optimizer.noise_multiplier))
                self.run.summary['noise'] = self.optimizer.noise_multiplier
            else: 
                logger.info('No Training with DP-SGD algorithm')
                self.run.summary['added_noise'] = 0

            self.criterion = nn.BCEWithLogitsLoss()
            self.sigmoid = nn.Sigmoid()
            self.sigmoid.to(self.device)
            self.model.to(self.device)

            
            logger.info('train() called: model={}, opt={}(lr={}), epochs={}, device={}'.format(
                    type(self.model).__name__, 
                    type(self.optimizer).__name__,
                    self.optimizer.param_groups[0]['lr'], 
                    self.config.epochs, 
                    self.device))

            logger.info('running with batch size {}'.format(self.config.batch_size))
        
            for self.epoch in range(1, self.config.epochs+1):
                # --- TRAIN ON TRAINING SET ------------------------------------------
                self.model.train()            
                self.train_loss = 0.0
                self.labels_all = torch.tensor((), device=self.device) # , device=torch.device('cuda:0')
                self.logits_prob_all = torch.tensor((), device=self.device)

                self.batch_loss = 0.0
                self.batch_count = 0
                self.step = 0
                wandb.watch(self.model, self.criterion, log="gradients")
                if (self.args.with_privacy):
                    with BatchMemoryManager(data_loader=self.train_dl, max_physical_batch_size = self.args.max_physical_batch_size, optimizer=self.optimizer) as new_data_loader:
                        self.batches_training(data_loader = new_data_loader)
                else: 
                    self.batches_training(data_loader = self.train_dl)      
                 
    def batches_training(self, data_loader):
        for batch_idx, (ecgs, ag, labels) in enumerate(data_loader,0):
            ecgs = ecgs.to(self.device) # ECGs
            ag = ag.to(self.device) # age and gender
            labels = labels.to(self.device) # diagnoses in SNOMED CT codes  
            #logger.info('input batch size {} '.format(len(labels)))

            with torch.set_grad_enabled(True):                    

                logits = self.model(ecgs, ag) 
                loss = self.criterion(logits, labels)
                logits_prob = self.sigmoid(logits)      
                loss_tmp = loss.item() * ecgs.size(0)
                self.labels_all = torch.cat((self.labels_all, labels), 0)
                self.logits_prob_all = torch.cat((self.logits_prob_all, logits_prob), 0)                    
                self.train_loss += loss_tmp
                
                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                # Printing training information
                if self.step % 5 == 0:
                    self.batch_loss += loss_tmp
                    self.batch_count += ecgs.size(0)
                    self.batch_loss = self.batch_loss / self.batch_count
                    logger.info('epoch {:^3} [{}/{}] train loss: {:>5.4f}'.format(
                        self.epoch, 
                        batch_idx * len(ecgs), 
                        len(self.train_dl.dataset), 
                        self.batch_loss
                    ))
                            
                    self.batch_loss = 0.0
                    self.batch_count = 0
                
            self.step += 1
            
        self.train_loss = self.train_loss / len(self.train_dl.dataset)            
        train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc, train_challenge_metric = cal_multilabel_metrics(self.labels_all, self.logits_prob_all, self.labels, self.args.threshold)
        logger.info('epoch {:^4}/{:^4} train loss: {:<6.2f}  train micro auroc: {:<6.2f}  train_macro_auroc: {:<6.2f}'.format( 
            self.epoch, 
            self.config.epochs, 
            self.train_loss, 
            train_micro_auroc,
            train_macro_auroc,
            train_macro_avg_prec,
            train_micro_avg_prec))  

        # LOGGING THE PERFORMANCE INTO WEIGHT AND BIAS
        wandb.log(
                    {
                        "train_micro_auroc": train_micro_auroc,\
                        "train_macro_auroc": train_macro_auroc,
                        "train_loss": self.train_loss,
                    }
                ) 
        
        if self.epoch == self.config.epochs:
            test_info = {
                'sph': {'macro_auroc': None, 'micro_auroc': None, 'roc_curves': None },
                'g12ec': {'macro_auroc': None, 'micro_auroc': None, 'roc_curves': None },
                'cpsc':  {'macro_auroc': None, 'micro_auroc': None, 'roc_curves': None },
                'ptb': {'macro_auroc': None, 'micro_auroc': None, 'roc_curves': None },
                'chapman': {'macro_auroc': None, 'micro_auroc': None, 'roc_curves': None }
            }
            if self.args.with_privacy:
                test_info['noise'] = self.optimizer.noise_multiplier
            else: 
                test_info['noise']= 0
                
            test_info['train_len'] = len(self.train_dl.dataset)
            
            # Running the Test Evaluation at the last epoch
            if self.args.test_paths != []:
                for test_path in self.args.test_paths:
                    # Load the test data
                    test_source_name, test_macro_auroc, test_micro_auroc, roc_curves_dic, classifiction_report_,\
                         y_true, y_pred, test_len = self.predict(test_path)
                    test_info['test_len'] = test_len
                    test_info[test_source_name]['macro_auroc'] = test_macro_auroc
                    test_info[test_source_name]['micro_auroc'] = test_micro_auroc
                    test_info[test_source_name]['roc_curves'] = roc_curves_dic
                    test_info[test_source_name]['classification_report'] = classifiction_report_
                    test_info[test_source_name]['y_true']= y_true
                    test_info[test_source_name]['y_pred']= y_pred


            self.run.summary['test_info'] = test_info

            #logger.info('start saving the best model')
            # model_path = wandb.run.dir+f"/{self.args.model_name}.onnx"
            #torch.onnx.export(self.model, (ecgs, ag),model_path)
            # torch.save(self.model, model_path)
            # wandb.save(model_path)

            
            logger.info('saving performance measurements in the run')
            path = wandb.run.dir+f"/{self.run.name}.pickle"
            with open(path, 'wb') as pickle_file:
                pickle.dump(test_info, pickle_file)
            wandb.save(path)
            #roc_curves_dic = roc_curves(self.labels_all, self.logits_prob_all, self.labels)

    def predict(self, test_path):  
        # Consider the GPU or CPU condition
        if test_path is not None:
                # Load the test data
                testing_set = ECGDataset(get_transforms('test'),
                                        target_df=None, path= test_path, amount = None)
                channels = testing_set.channels
                self.test_dl = DataLoader(testing_set,
                                        batch_size=1,
                                        shuffle=False,
                                        pin_memory=(True if self.device == 'cuda' else False),
                                        drop_last=True)
                
        test_source_name = test_path.split('.')[0].split('/')[-1].split('_')[-1]
        logger.info('test source name is: {}'.format(test_source_name))
        logger.info('total test samples is: {}'.format(len(testing_set)))

        #self.model =  onnx.load(self.args.model_path)
        # self.model = convert(self.args.model_path)
        # print(type(self.model))

        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)
        
        self.model.eval()
        self.labels_all = torch.tensor((), device=self.device)
        self.logits_prob_all = torch.tensor((), device=self.device)  
        
        for i, (ecgs, ag, labels) in enumerate(self.test_dl):
            ecgs = ecgs.to(self.device) # ECGs
            ag = ag.to(self.device) # age and gender
            labels = labels.to(self.device) # diagnoses in SMONED CT codes 

            with torch.set_grad_enabled(False):  
                
                logits = self.model(ecgs, ag)
                logits_prob = self.sigmoid(logits)
                self.labels_all = torch.cat((self.labels_all, labels), 0)
                self.logits_prob_all = torch.cat((self.logits_prob_all, logits_prob), 0)

        
            # ------ One-hot-encode predicted label -----------
            # Define an empty label for predictions
            pred_label = np.zeros(len(self.labels))

            # Find the maximum values within the probabilities
            _, likeliest_dx = torch.max(logits_prob, 1)

            # Predicted probabilities from tensor to numpy
            likeliest_dx = likeliest_dx.cpu().detach().numpy()

            # First, add the most likeliest diagnosis to the predicted label
            pred_label[likeliest_dx] = 1

            # Then, add all the others that are above the decision threshold
            other_dx = logits_prob.cpu().detach().numpy() >= self.args.threshold
            pred_label = pred_label + other_dx
            pred_label[pred_label > 1.1] = 1
            pred_label = np.squeeze(pred_label)

            # --------------------------------------------------
            
            # Save also probabilities but return them first in numpy
            scores = logits_prob.cpu().detach().numpy()
            scores = np.squeeze(scores)
            
            # Save the prediction
            #self.save_predictions(self.filenames[i], pred_label, scores, self.args.pred_save_dir)

            if i % 1000 == 0:
                logger.info('{:<4}/{:>4} predictions made'.format(i+1, len(self.test_dl)))

        # Predicting metrics
        test_macro_avg_prec, test_micro_avg_prec, test_macro_auroc, test_micro_auroc, test_challenge_metric = cal_multilabel_metrics(self.labels_all, self.logits_prob_all, self.labels, self.args.threshold) 
        logger.info('macro avg prec: {:<6.2f} micro avg prec: {:<6.2f} macro auroc: {:<6.2f} micro auroc: {:<6.2f} challenge metric: {:<6.2f}'.format(
            test_macro_avg_prec,
            test_micro_avg_prec,
            test_macro_auroc,
            test_micro_auroc,
            test_challenge_metric))

        classifiction_report_ = cal_classification_report(self.labels_all, self.logits_prob_all, self.labels)
        # Draw ROC curve for predictions
        roc_curves_dic = roc_curves(self.labels_all, self.logits_prob_all, self.labels)

        test_len = len(self.test_dl)    
        return test_source_name, test_macro_auroc, test_micro_auroc, roc_curves_dic,classifiction_report_ ,\
            self.labels_all, self.logits_prob_all,test_len
        
# function to remove unknown genders and negative and zero ages

def pre_processing(df):
    df['gender'] = df['gender'].apply(lambda x: 'Male' if x=='M' else 'Female' if x=='F' else x)
    df = df[(df['gender'] =='Male') | (df['gender'] == 'Female')] # filter the rows where the genders are not known
    df = df[df['age']>0] # filtering the rows which has negative ages
    return df

# Reading the G12EC
CPSC_FOLDER_NAME = 'CPSC_CPSC_Extra'
G12EC_FOLDER_NAME = 'G12EC'
PTB_FOLDER_NAME = 'PTB_PTBXL'
SPH_FOLDER_NAME = 'SPH'
CHAPMAN_FOLDER_NAME = 'ChapmanShaoxing_Ningbo'

test_paths = [
    #os.path.join(parent_path, 'data', 'split_csvs', CPSC_FOLDER_NAME,'clean_all_cpsc.csv'),
    os.path.join(parent_path, 'data', 'split_csvs', G12EC_FOLDER_NAME,'clean_all_g12ec.csv'),
    # os.path.join(parent_path, 'data', 'split_csvs', PTB_FOLDER_NAME,'clean_all_ptb.csv'),
    # os.path.join(parent_path, 'data', 'split_csvs', SPH_FOLDER_NAME,'clean_all_sph.csv'),
    # os.path.join(parent_path, 'data', 'split_csvs', CHAPMAN_FOLDER_NAME,'clean_all_chapman.csv')
]

four_hospitals_train = pd.concat([
        pd.read_csv('/home/...../data/split_csvs/ChapmanShaoxing_Ningbo/clean_all_chapman.csv'),
        pd.read_csv('/home/...../data/split_csvs/CPSC_CPSC_Extra/clean_all_cpsc.csv'),
        pd.read_csv('/home/.....data/split_csvs/PTB_PTBXL/clean_all_ptb.csv'),
        pd.read_csv('/home/...../data/split_csvs/SPH/clean_all_sph.csv')
])
four_hospitals_train.to_csv('/home/....../data/split_csvs/clean_cpsc_sph_chapman_ptb.csv',index=False)

if __name__ == '__main__':
    eps  = sys.argv[1] 
    if eps != 'inf':
        eps = int(eps) 
    epo =  int(sys.argv[2])
    batch_size  =  int(sys.argv[3])
    model_name = sys.argv[4] # e.g resnet18_v1
    model_version = sys.argv[5] # v1
    gpu_idx = sys.argv[6] # 1
    base_seed = int(sys.argv[7]) # 1

    random.seed(base_seed)

    # Generate ten random seeds
    random_seeds = [random.randint(0, 2**32 - 1) for _ in range(10)]

    # Print the generated seeds
    for i, seed in enumerate(random_seeds):
        print(f"Seed {i + 1}: {seed}")
        best_private_config= {
                'train_path': os.path.join(parent_path, 'data', 'split_csvs', 'clean_cpsc_sph_chapman_ptb.csv'),
                'val_path': None,
                'test_paths': test_paths,
                'num_workers': 1,
                'weight_decay': 0.000010,
                'device_count': 1,
                'threshold': 0.500000,
                'with_privacy': True if eps!='inf' else False,
                'accountant':'rdp',
                'max_physical_batch_size' :100,
                'project_name': 'Experiments_with_10_fixed_random_seeds',
                'model_name': model_name,    # change here
                'sweep_count': None,    
                'source_name': 'cpsc_sph_chapman_ptb',
                'epsilon': eps,              # change here
                'gpu_index':'0',               # change here
                'model_version': model_name ,# change here
                'random_seed': seed
                
        }
        best_private_sweep_config = { 
            'name' : f'best_private_eps_{eps}_{model_version}', # change here
            'method':'grid',
            'metric': {
                'name': 'val_loss',
                'goal': 'minimize'
            },
            'parameters':
            {
                'batch_size': {
                    #'values':[8,64,400,600,800]
                    'value': batch_size
                },
                'epochs':{
                    'value': epo
                },
                'epsilon' : {
                    'value': eps  # change here
                },
                'max_grad_norm' : {
                    'value': 1
                },
                'lr': {
                    'value': 0.003
                },
                'delta': {
                    'value':1 / len(four_hospitals_train)
                }
            }
            
            }

        best_private_config = dict2obj(best_private_config)
        best_private_trainer = Training_WANDB(best_private_config, best_private_sweep_config)
        best_private_trainer.setup()
        best_private_trainer.train_with_sweep()