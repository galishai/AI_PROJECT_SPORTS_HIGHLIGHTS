import torch
import torch.nn as nn
import torch.optim as optim
#from training import *
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import argparse
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import tqdm

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def time_to_seconds(time_str):
    if ':' in time_str:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    else:
        return int(float(time_str))

def seconds_to_time(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:02d}"


def get_dataset(path, verbose=False):
    dataset = pd.read_csv(path)
    aug_count = 0
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1
    #quarter to categorical
    dataset = pd.get_dummies(dataset, columns=['quarter'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #time to ordinal
    dataset['time_left_qtr'] = dataset['time_left_qtr'].apply(time_to_seconds)
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #dates to ordinal
    dataset['date'] = pd.to_datetime(dataset['date'], format='%B %d, %Y')
    first_date = dataset['date'].min()
    dataset['days_since_first_game'] = (dataset['date'] - first_date).dt.days
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1
    dataset = dataset.drop(columns=['date'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #convert plays to categorical
    dataset = pd.get_dummies(dataset, columns=['play'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #teams to categorical
    dataset = pd.get_dummies(dataset, columns=['home_team', 'away_team', 'current_team'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #player names to categorical
    dataset = pd.get_dummies(dataset, columns=['name', 'assister', 'stolen_by'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    return dataset


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

        self.relu = nn.ReLU()

        self.batchnorm1 = nn.BatchNorm1d(2 * input_dim)
        self.batchnorm2 = nn.BatchNorm1d(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # single logit for BCE
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            ys.extend(yb.cpu().numpy())
            ps.extend(probs.cpu().numpy())
    ys, ps = np.array(ys), np.array(ps)
    preds = (ps >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(ys, preds),
        'precision': precision_score(ys, preds, zero_division=0),
        'recall': recall_score(ys, preds, zero_division=0),
        'f1': f1_score(ys, preds, zero_division=0),
        'roc_auc': roc_auc_score(ys, ps)
    }

def train_fold(model, train_loader, val_loader, device, epochs, lr, threshold):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_f1 = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        pbar_name = "batch_num"
        with tqdm.tqdm(desc=pbar_name, total=len(train_loader)) as pbar:
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                pbar.update(1)

            val_metrics = evaluate(model, val_loader, device, threshold)
            pbar.set_description(f"{pbar_name} Epoch {epoch:02d} | Threshold {threshold:.2f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | "
                f"Precision: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"AUC: {val_metrics['roc_auc']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = model.state_dict()
    model.load_state_dict(best_state)
    return model


def main():



    data = "../full season data/plays_with_onehot_v2.csv"

    folds = 0
    epochs = 20
    batch_size = 64
    hidden_dim = 256
    dropout = 0.5
    lr = 1e-03
    seed = 42
    threshold = 0.5
    grid_search = False

    freeze_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print('Using device:', device)

    df = get_dataset(data)

    X = df.drop(columns=['is_highlight']).values.astype(np.float32)
    y = df['is_highlight'].values.astype(int)

    if folds > 0:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        all_fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            print(f"===== Fold {fold}/{folds} =====")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

            model = MLPClassifier(input_dim=X.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
            model = train_fold(model, train_loader, val_loader, device, epochs, lr=lr, threshold=threshold)
            metrics = evaluate(model, val_loader, device, threshold=threshold)
            all_fold_metrics.append(metrics)

        # Aggregate
        agg = {k: np.mean([m[k] for m in all_fold_metrics]) for k in all_fold_metrics[0]}
        print("===== Cross-Validation Results =====")
        for k,v in agg.items(): print(f"{k}: {v:.4f}")
    else:
        X_train, X_val ,y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = MLPClassifier(input_dim=X.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
        model = train_fold(model, train_loader, val_loader, device, epochs, lr=lr, threshold=threshold)
        metrics = evaluate(model, val_loader, device, threshold=threshold)

        print("===== Training Results =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    # Final train on full data & test (if you have a held-out test set)

if __name__ == '__main__':
    main()

'''
if __name__ == "__main__":
    # Hyperparameters
    input_size = 784  # Example: 28x28 flattened image
    hidden_size = 256  # Example hidden size
    output_size = 10  # Example: number of classes (e.g., MNIST)
    batch_size = 32

    file_path = "../full season data/plays_with_onehot_v1.csv"

    frozen_seed = 42

    freeze_seeds(frozen_seed)

    nba_dataset = get_dataset(file_path)

    train_dataset, test_dataset = train_test_split(nba_dataset, test_size=0.2, shuffle=True,
                                                   random_state=frozen_seed)

    X_train = train_dataset.drop(columns=['is_highlight'])
    y_train = train_dataset['is_highlight']
    # X_val = val_dataset.drop(columns=['is_highlight'])
    # y_val = val_dataset['is_highlight']
    X_test = test_dataset.drop(columns=['is_highlight'])
    y_test = test_dataset['is_highlight']


    mlp_classifier = Classifier()
'''
