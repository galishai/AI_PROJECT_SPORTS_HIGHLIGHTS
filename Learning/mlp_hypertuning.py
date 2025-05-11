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
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer, fbeta_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import tqdm
import copy


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

    return preds

def compute_validation_loss(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train_fold(model, train_loader, val_loader, device, epochs, lr, threshold, patience=5, min_delta=1e-5):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batch_count = 0
        pbar_name = f"epoch: {epoch}"
        with tqdm.tqdm(desc=pbar_name, total=len(train_loader)) as pbar:
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device).float()
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({
                    'batch': batch_idx,
                    'loss': loss.item(),
                })

                pbar.update(1)
        avg_train_loss = epoch_loss / batch_count
        avg_val_loss = compute_validation_loss(model, val_loader, criterion, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch} | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if patience is not None and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_model is not None:
        model.load_state_dict(best_model)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='x', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig('loss_curves.png')
    plt.show()
    return model


def main():
    data = "../full season data/plays_with_onehot_v2.csv"
    is_tuning = False
    folds = 0
    epochs = 20
    batch_size = 512
    hidden_dim = 128
    dropout = 0.3
    lr = 1e-03
    seed = 42
    threshold = 0.2
    betas = [0.5, 1.0, 2.0]

    freeze_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print('Using device:', device)

    df = get_dataset(data)

    X = df.drop(columns=['is_highlight']).values.astype(np.float32)
    y = df['is_highlight'].values.astype(int)

    if folds <= 0:
        if not is_tuning:
            param_grid = {
                'hidden_dim': [hidden_dim],
                'dropout': [dropout],
                'lr': [lr],
                'threshold': [threshold],
                'epochs': [epochs],
                'batch_size': [batch_size],
            }
        else:
            param_grid = {
                'hidden_dim': [128, 256],
                'dropout': [0.3, 0.5],
                'lr': [1e-3, 1e-4],
                'threshold': [0.1, 0.15,0.2,0.25],
                'epochs': [10],
                'batch_size': [256, 512],
            }
        X_train, X_val ,y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        results = []
        for params in ParameterGrid(param_grid):
            print(params)
            model = MLPClassifier(input_dim=X.shape[1], hidden_dim=params['hidden_dim'], dropout=params['dropout']).to(device)
            model = train_fold(model, train_loader, val_loader, device, params['epochs'], params['lr'], params['threshold'])
            y_pred = evaluate(model, val_loader, device, threshold=params['threshold'])
            record = {**params}
            for β in betas:
                record[f'f_beta_{β}'] = fbeta_score(y_val,
                                                    y_pred,
                                                    beta=β,
                                                    zero_division=0)
            record['y_pred'] = y_pred
            results.append(record)

        best_per_beta = {}
        for β in betas:
            best_per_beta[β] = max(
                results,
                key=lambda rec: rec[f'f_beta_{β}']
            )
        for β, best in best_per_beta.items():
            print(f"\n=== Results for β={β} (F{β}-score) ===")
            print(f"  → Best F{β}-score: {best[f'f_beta_{β}']:.4f}")
            print("  → Hyper-parameters:")
            for h in ['hidden_dim', 'dropout', 'lr', 'threshold', 'epochs']:
                print(f"      • {h}: {best[h]}")

            y_pred = best['y_pred']

            # confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            print("\n  Confusion Matrix:")
            print(cm)

            # classification report (precision, recall, f1 for each class + averages)
            print("\n  Classification Report:")
            print(classification_report(
                y_val, y_pred, zero_division=0, digits=4
            ))

        return 0

    else:
        raise NotImplemented;

if __name__ == '__main__':
    main()

