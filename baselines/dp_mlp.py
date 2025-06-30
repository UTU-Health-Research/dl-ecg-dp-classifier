import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from utils import preprocess_data
from opacus.accountants import RDPAccountant
import math

# Check for required dependencies
try:
    import numpy as np
except ImportError:
    print("Error: NumPy is not installed. Please install it using: pip install numpy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: Pandas is not installed. Please install it using: pip install pandas")
    sys.exit(1)

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("Error: scikit-learn is not installed. Please install it using: pip install scikit-learn")
    sys.exit(1)

try:
    from opacus import PrivacyEngine
except ImportError:
    print("Error: Opacus is not installed. Please install it using: pip install opacus")
    sys.exit(1)

def calculate_noise_multiplier(epsilon, delta, epochs, batch_size, dataset_size):
    """
    Calculate the noise multiplier based on privacy budget and training parameters
    
    Args:
        epsilon: Privacy budget
        delta: Privacy parameter
        epochs: Number of training epochs
        batch_size: Batch size
        dataset_size: Total number of training samples
    """
    # Calculate sampling probability
    sampling_probability = batch_size / dataset_size
    
    # Initialize RDP accountant
    accountant = RDPAccountant()
    
    # Calculate noise multiplier using RDP analysis
    # This is a simplified version - in practice, you might want to use more sophisticated methods
    # or use Opacus's built-in methods for calculating noise multiplier
    noise_multiplier = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise_multiplier *= math.sqrt(epochs * sampling_probability)
    
    return noise_multiplier

class ECGClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ECGClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train_dp_model(X_train, y_train, X_test, y_test, 
                  epsilon=1.0, delta=1e-5, 
                  batch_size=32, epochs=10, 
                  learning_rate=0.001):
    """
    Train a differentially private neural network model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        epsilon: Privacy budget
        delta: Privacy parameter
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]
    model = ECGClassifier(input_dim, num_classes)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Calculate noise multiplier based on privacy budget
    noise_multiplier = calculate_noise_multiplier(
        epsilon=epsilon,
        delta=delta,
        epochs=epochs,
        batch_size=batch_size,
        dataset_size=len(X_train)
    )
    
    print(f"Calculated noise multiplier: {noise_multiplier:.4f}")
    
    # Initialize privacy engine
    privacy_engine = PrivacyEngine()
    
    # Make model and optimizer private
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=1.0,
    )
    
    # Training loop
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Print training progress
        avg_loss = total_loss / len(train_loader)
        epsilon_used = privacy_engine.get_epsilon(delta)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Epsilon: {epsilon_used:.2f}')
        
        # Early stopping if we exceed privacy budget
        if epsilon_used > epsilon:
            print(f'Privacy budget exceeded. Stopping training at epoch {epoch+1}')
            break
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_outputs = test_outputs.numpy()
        
        # Calculate ROC-AUC scores
        roc_auc_macro = roc_auc_score(y_test, test_outputs, average='macro')
        roc_auc_micro = roc_auc_score(y_test, test_outputs, average='micro')
        
        print(f'Final Macro-average ROC-AUC: {roc_auc_macro:.4f}')
        print(f'Final Micro-average ROC-AUC: {roc_auc_micro:.4f}')
    
    return model, privacy_engine

if __name__ == "__main__":
    # Load and preprocess data
    train_sources = ['chapman', 'cpsc', 'ptb', 'sph']
    test_source = 'g12ec'
    
    # Load training data
    train_ids = []
    for source in train_sources:
        df = pd.read_csv(f'data/sources_clean_zoher/clean_all_{source}.csv')
        for idx, row in df.iterrows():
            train_ids.append(row['path'].split('/')[-1].split('.')[0].split('_')[0])
    
    # Load test data
    test_ids = []
    df = pd.read_csv(f'data/sources_clean_zoher/clean_all_{test_source}.csv')
    for idx, row in df.iterrows():
        test_ids.append(row['path'].split('/')[-1].split('.')[0].split('_')[0])
    
    # Load features
    train_feats = pd.DataFrame()
    for source in ['ChapmanShaoxing_Ningbo', 'CPSC_CPSC-Extra', 'PTB_PTBXL', 'SPH']:
        source_feats = pd.read_csv(f'data/{source}_feats.csv')
        train_feats = pd.concat((train_feats, source_feats), ignore_index=True)
    train_feats = train_feats[train_feats['id'].isin(train_ids)]
    
    df_test = pd.read_csv('data/G12EC_feats.csv')
    test_feats = df_test[df_test['id'].isin(test_ids)]
    
    # Preprocess data
    X_train, y_train = preprocess_data(train_feats)
    X_test, y_test = preprocess_data(test_feats)
    
    # Train model with different privacy budgets
    privacy_budgets = [0.5, 1.0, 5.0, 10.0]
    
    for epsilon in privacy_budgets:
        print(f"\nTraining with privacy budget epsilon = {epsilon}")
        model, privacy_engine = train_dp_model(
            X_train, y_train, X_test, y_test,
            epsilon=epsilon,
            delta=1e-5,
            batch_size=32,
            epochs=10,
            learning_rate=0.001
        ) 
