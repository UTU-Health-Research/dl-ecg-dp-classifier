import pandas as pd
import os
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

def predict_all_diagnoses(multi_diag_model, X, y=None):
    predictions = {}
    for column, model in multi_diag_model.items():
        predictions[column.split('_')[-1]] = model.predict_proba(X)[:, 1]

    predictions = pd.DataFrame(predictions)
    if y is not None:
        y_true_flat = y.values.ravel()
        y_pred_flat = predictions.values.ravel()
        micro_auc = roc_auc_score(y_true_flat, y_pred_flat)

        auc_scores = []
        for column in y.columns:
            auc = roc_auc_score(y[column], predictions[column])
            auc_scores.append(auc)
        macro_auc = np.mean(auc_scores)

        return predictions, micro_auc, macro_auc
    
    return predictions

def plot_feature_importances(importances, feature_names, diagnosis):
    indices = np.argsort(importances)
    plt.title(f'Feature Importances for {diagnosis}')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')

def macro_avg_accuracy(y_true, y_pred):
    accuracies = []
    for i in range(y_true.shape[1]):
        accuracy = balanced_accuracy_score(y_true[:, i], y_pred[:, i])
        accuracies.append(accuracy)
    return np.mean(accuracies)
    

def mdm_pred_prob(multi_diag_model, X):
    y_probs = {}
    for k in multi_diag_model.keys():
        y_probs[k] = multi_diag_model[k][0].predict_proba(X[multi_diag_model[k][1]])[:,1]
    
    return pd.DataFrame(y_probs)



def preprocess_data(file='data/ChapmanShaoxing_Ningbo_feats.csv', test_size=0, scaler=True, random_state=42, print_percentages=False):

    if type(file) == pd.DataFrame:
        df = file
    else:
        df = pd.read_csv(file)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        df.replace('#NAME?', np.nan, inplace=True)
        df['swt_d_3_energy_entropy'] = df['swt_d_3_energy_entropy'].astype(float)
        df['swt_d_4_energy_entropy'] = df['swt_d_4_energy_entropy'].astype(float)

        features = df[['age', 'sex', 'heart_rate_min', 't_wave_multiscale_permutation_entropy_std', 'heart_rate_max', 't_wave_multiscale_permutation_entropy_median',
                    'rs_time_std', 'p_wave_corr_coeff_median', 'rri_median', 'heart_rate_mean', 'rri_cluster_ssd_3', 'rri_fisher_info', 'pnn60',
                    'swt_d_4_energy_entropy', 'rri_cluster_ssd_2', 'heart_rate_activity', 'diff_rri_min', 't_wave_permutation_entropy_std',
                    'p_wave_sample_entropy_std', 'swt_d_3_energy_entropy', 'p_wave_approximate_entropy_median', 'rpeak_approximate_entropy']]

        y = df[['270492004', '164889003', '164890007', '713426002', '445118002', '39732003', '164909002', '251146004', '284470004',
                        '47665007', '59118001', '427393009', '426177001', '426783006', '427084000', '164934002', '59931005']]

        X = features.fillna(features.median()).replace([np.inf, -np.inf], 0)

        if scaler:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        X_df = pd.DataFrame(X, columns=features.columns)

        if test_size == 0:
            return X_df, y

        stratify_vector = y.sum(axis=1)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        for train_index, test_index in splitter.split(X_df, stratify_vector):
            X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if print_percentages:
            for c in y.columns:
                test_ratio = np.sum(y_test[c])/np.sum(y_train[c])*100
                print(f'{c}: {100-test_ratio:.0f}/{test_ratio:.0f}')

        return X_train, X_test, y_train, y_test
    
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_private_logreg(X_train, y_train, epsilon, num_epochs=500, lr=20, batch_size=50000, delta=None, max_grad_norm=1, print_epoch_information=True):
    
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(X_train.columns)
    output_dim = len(y_train.columns)

    model = LogisticRegressionModel(input_dim, output_dim)

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if delta == None:
        delta = 1/len(X_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        privacy_engine = PrivacyEngine(accountant='rdp')

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=num_epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )

        for epoch in range(num_epochs):

            model.train()

            for X_batch, y_batch in train_loader:
                
                y_predicted = model(X_batch)
                loss = criterion(y_predicted, y_batch)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            if print_epoch_information:
                epsilon = privacy_engine.get_epsilon(delta=1/len(X_train))
                print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Epsilon: {epsilon:.2f}', end='\r')

    return model

def train_logreg(X_train, y_train, num_epochs=500, lr=20, batch_size=50000, print_epoch_information=True):
    
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(X_train.columns)
    output_dim = len(y_train.columns)

    model = LogisticRegressionModel(input_dim, output_dim)

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        model.train()

        for X_batch, y_batch in train_loader:
            
            y_predicted = model(X_batch)
            loss = criterion(y_predicted, y_batch)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if print_epoch_information:
            print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}', end='\r')

    return model


def test_private_logreg(model, X_test, y_test, labels=None):

    X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)

    if labels == None:
        labels = y_test.columns
        
    label_inds = [y_test.columns.get_loc(l) for l in labels]
    y_test = y_test[labels]

    model.eval()

    with torch.no_grad():
        y_pred_prob = model(X_test_tensor)[:, label_inds]

    if len(labels) == 1:
        return round(roc_auc_score(y_test, y_pred_prob), 4)
    
    else:
        return {
            'macro_auc': round(roc_auc_score(y_test, y_pred_prob, average='macro'), 4),
            'micro_auc': round(roc_auc_score(y_test, y_pred_prob, average='micro'), 4)
        }


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


def train_private_logreg_with_mask(X_train, y_train, epsilon, mask=None, num_epochs=500, lr=20, batch_size=50000, delta=None, max_grad_norm=1, print_epoch_information=True):
    
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    class LogisticRegressionModelMasked(nn.Module):
        def __init__(self, input_dim, output_dim, mask=None):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.mask = mask

            if mask is not None:
                if isinstance(mask, int):
                    mask_ = torch.zeros(output_dim, input_dim)
                    mask_[:, np.random.choice(input_dim, mask, replace=False)] = 1
                    self.mask = mask_
                else:
                    self.mask = torch.Tensor(mask, dtype=torch.float32)
        
        def forward(self, x):
            masked_weights = self.linear.weight*self.mask
            z = torch.matmul(x, masked_weights.t()) + self.linear.bias
            return torch.sigmoid(z)
    
    input_dim = len(X_train.columns)
    output_dim = len(y_train.columns)

    model = LogisticRegressionModel(input_dim, output_dim, mask)

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if delta == None:
        delta = 1/len(X_train)

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")

    privacy_engine = PrivacyEngine(accountant='rdp')

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=num_epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    for epoch in range(num_epochs):

        model.train()

        for X_batch, y_batch in train_loader:
            
            y_predicted = model(X_batch)
            loss = criterion(y_predicted, y_batch)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if print_epoch_information:
            epsilon = privacy_engine.get_epsilon(delta=1/len(X_train))
            print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Epsilon: {epsilon:.2f}', end='\r')

    return model
