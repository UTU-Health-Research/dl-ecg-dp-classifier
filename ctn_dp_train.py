import os
import pandas as pd
import numpy as np
import logging
import torch
import random
from utils import load_yaml
from src.modeling.train_utils_dp import Training

# Set seed
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ======== MANUAL CONFIG START ======== #
for seed in range(11, 12):
    class Args:
        pass

    args = Args()
    
    args.feature_path = 'smoke_features'
    args.csv_path = 'stratified_smoke'
    args.train_file = 'train_split_1_1.csv'
    args.val_file = 'val_split_1_1.csv'
    args.yaml_file_name = 'manual_config'


    #feature_root = os.path.join(os.getcwd(), 'data', 'features', args.feature_path)
    csv_root = os.path.join(os.getcwd(), 'data', 'split_csvs', args.csv_path)
    args.train_path = '..../data/clean_cpsc_sph_chapman_ptb.csv'
    args.val_path =  None
    # Add test path for evaluation
    args.test_path = '...r/tansformer/data/clean_all_g12ec.csv'

    args.model_save_dir = os.path.join(os.getcwd(), 'experiments', args.yaml_file_name)
    args.roc_save_dir = os.path.join(args.model_save_dir, 'ROC_curves')

    # ======== MANUAL CONFIG END ======== #

    args.labels = pd.read_csv(args.train_path, nrows=0).columns.tolist()[4:]

    
    args.all_features = None
    

    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.roc_save_dir, exist_ok=True)

    logs_path = os.path.join(args.model_save_dir, 'manual_config_train_dp.log')
    logging.basicConfig(filename=logs_path,
                        format='%(asctime)s %(message)s',
                        filemode='w',
                        datefmt='%Y-%m-%d %H:%M:%S')
    args.logger = logging.getLogger(__name__)
    args.logger.setLevel(logging.DEBUG)

    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger.info(f'Device: {args.device}')
    args.device_count = 1
    if args.device_count > 1:
        args.logger.info(f'Using {args.device_count} GPUs')
    else:
        args.logger.info('Using 1 GPU or CPU')

    args.batch_size = 400
    args.epochs = 5
    args.num_workers = 0
    args.lr = 0.003
    args.weight_decay = 1e-05
    args.threshold = 0.5
    args.seed = seed
    # Differential Privacy arguments
    args.dp = True
    args.epsilon = 1
    args.delta = 1 / len(pd.read_csv(args.train_path))
    args.max_grad_norm = 0.1
    args.accountant = 'rdp'

    # Set environment variable to manage CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    print('max_grad_norm:', args.max_grad_norm)
    print('epsilon:', args.epsilon)

    # Train
    print('Starting training...')
    args.logger.info('Starting training...')
    trainer = Training(args)
    trainer.setup()
    trainer.train()
    print('Training complete.')
    
    # Clean up memory after training for this seed
    print(f'Cleaning up memory after seed {seed}...')
    args.logger.info(f'Cleaning up memory after seed {seed}...')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release all unused memory back to the OS
        torch.cuda.synchronize()  # Ensure all operations are complete before proceeding
    del trainer  # Delete the trainer object to free up memory
    print(f'Memory cleanup complete for seed {seed}.')
    args.logger.info(f'Memory cleanup complete for seed {seed}.')


 
