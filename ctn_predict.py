import os
import pandas as pd
import numpy as np
import torch
import random
import logging
from src.modeling.predict_utils import Predicting

# Set seed for reproducibility
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Define configuration directly in the script
class Args:
    pass

args = Args()

# Set your specific paths and filenames here
args.csv_path = 'stratified_smoke'  # Subfolder inside data/split_csvs
args.test_file = 'test_split_1_1.csv'  # Test CSV file name
args.model = 'manual_config_dp.pth'  # Name of the trained model file
args.yaml_file_name = 'manual_config'  # Base name for output directory and logs

# Construct derived paths
csv_root = os.path.join(os.getcwd(), 'data', 'split_csvs', args.csv_path)
args.test_path = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/G12EC/clean_all_g12ec.csv'
args.output_dir = os.path.join(os.getcwd(), 'experiments', args.yaml_file_name)

# Locate the trained model
model_path = os.path.join(args.output_dir, args.model)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")
args.model_path = model_path

# Load labels from the test CSV
args.labels = pd.read_csv(args.test_path, nrows=0).columns.tolist()[4:]

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Set up logging
logs_path = os.path.join(args.output_dir, args.yaml_file_name + '_predict.log')
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

args.threshold = 0.5

# Log arguments
args.logger.info('Arguments:')
args.logger.info('-'*10)
for k, v in args.__dict__.items():
    args.logger.info(f'{k}: {v}')
args.logger.info('-'*10)

args.logger.info('Making predictions...')

# Set device
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.logger.info(f'Device: {args.device}')

# Initialize and run prediction
pred = Predicting(args)
pred.setup()
pred.predict()

print('Predictions completed.')