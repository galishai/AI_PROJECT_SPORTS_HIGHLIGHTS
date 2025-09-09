from sched import scheduler

import torch
import torch.nn as nn
import numpy as np
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm_notebook as tqdm
import copy
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torchmetrics import F1Score
from sklearn.metrics import average_precision_score



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = probability of true class
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()

        layers = []
        prev = input_dim
        assert(len(hidden_dims) > 0)

        for d in hidden_dims:
            layers += [
                nn.Linear(prev, d),
                nn.BatchNorm1d(d),
                nn.ReLU(),
                nn.Dropout(dropout)
                ]
            prev = d
        layers.append(nn.Linear(prev,1))

        self.net = nn.Sequential(*layers)
        '''self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)  # single logit for BCE
        )'''

    def forward(self, x):
        return self.net(x).squeeze(1)

def evaluate(model, loader, device, threshold=0.5):

    y_true, y_probs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            y_true.extend(yb.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)



    return y_true, y_probs

def compute_validation_loss(model, val_loader, device, epoch):
    val_metric = F1Score(task="binary", threshold=0.5).to(device)

    critereon_val = nn.BCEWithLogitsLoss()
    model.eval()
    val_metric.reset()

    val_loss = 0.0
    y_true, y_probs = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            logits = model(xb)
            probs = torch.sigmoid(logits)
            val_metric.update(probs, yb.int())
            loss = critereon_val(logits, yb)
            val_loss += loss.item()
            y_true.extend(yb.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    val_f1 = val_metric.compute().item()
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    val_ap = average_precision_score(y_true, y_probs)
    print(f"Epoch {epoch} | Val F1: {val_f1:.4f} | Val AP: {val_ap:.4f}")
    return val_loss / len(val_loader), val_f1, val_ap

def train_fold(model, train_loader, val_loader, device, epochs, lr, pos_weight=0, patience=10, min_delta=1e-5, save_model=False, rm_ft_ds=False, weight_decay = 0, objective='f1', lr_warmup_func = None):
    if pos_weight.item() != 0:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        #print("using bce")
    else:
        criterion = FocalLoss(alpha=0.15, gamma=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(optimizer.param_groups[0]['lr'])
    print(optimizer.param_groups[0]['weight_decay'])

    if objective == 'auprc' or objective == 'f1':
        sched_mode= 'max'
    else:
        sched_mode = 'min'
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode = 'min', #sched_mode
        #factor=0.5,
        #patience=3,
        #cooldown=0,
        #min_lr=1e-6,
        #verbose= True
    )


    '''steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=1e2)'''
    model.train()
    train_losses ,val_losses = [] , []
    train_f1_scores, val_f1_scores, val_ap_scores = [], [], []
    best_val_loss = float('inf')
    best_val_f1 = -float('inf')
    best_val_ap = -float('inf')
    best_model_state_dict = None
    optimizer_best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        '''if lr_warmup_func is None:
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=3,
                # min_lr=1e-6
            )
        else:
            scheduler = None
            lr = lr_warmup_func(epoch, lr)'''
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr
        epoch_loss = 0.0
        batch_count = 0
        train_metric = F1Score(task="binary", threshold=0.5).to(device)
        train_metric.reset()
        pbar_name = f"epoch: {epoch}"
        with tqdm(desc=pbar_name, total=len(train_loader)) as pbar:
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device).float()
                optimizer.zero_grad()
                logits = model(xb)
                probs = torch.sigmoid(logits)
                train_metric.update(probs, yb.int())
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                #scheduler.step()

                epoch_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({
                    'batch': batch_idx,
                    'loss': loss.item(),
                })
                pbar.update(1)
        train_f1 = train_metric.compute().item()
        #print(f"Epoch {epoch} | Train F1: {train_f1:.4f}")
        train_f1_scores.append(train_f1)
        avg_train_loss = epoch_loss / batch_count

        avg_val_loss, val_f1, val_ap = compute_validation_loss(model, val_loader, device, epoch)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_f1_scores.append(val_f1)
        val_ap_scores.append(val_ap)
        '''if scheduler is not None:
            if objective == 'auprc':
                scheduler.step(1-val_ap)
            elif objective == 'f1':
                scheduler.step(1 - val_f1)
            else:
                scheduler.step(avg_val_loss)'''

        print(f"Epoch {epoch} | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")
        #scheduler.step(avg_val_loss)
        if objective == 'auprc':
            #scheduler.step(val_ap)
            tracked_score = val_ap

            if "best_val_ap" not in locals():
                best_val_ap = -float('inf')
            improved = (val_ap > best_val_ap - min_delta)
        elif objective == 'f1':
            #scheduler.step(val_f1)
            tracked_score = val_f1

            if "best_val_f1" not in locals():
                best_val_f1 = -float('inf')
            improved = (val_f1 > best_val_f1 - min_delta)
        else: #loss
            #scheduler.step(avg_val_loss)
            tracked_score = -avg_val_loss

            if "best_val_loss" not in locals():
                best_val_loss = float('inf')
            improved = (avg_val_loss < best_val_loss - min_delta)

        if improved:
            if objective== "auprc":
                best_val_ap = val_ap
            elif objective == "f1":
                best_val_f1 = val_f1
            else:
                best_val_loss = avg_val_loss

            best_model_state_dict = copy.deepcopy(model.state_dict())
            optimizer_best_state_dict = copy.deepcopy(optimizer.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience is not None:
                print(f"no {objective} improvement for {epochs_without_improvement} epoch(s).")
            if patience is not None and epochs_without_improvement >= patience:
                print(f"early stopping at epoch {epoch}.")
                break

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        if save_model:
            save_path = "saved_model/mlp_tuned.pth"
            torch.save({
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer_best_state_dict,
            }, save_path)
            print(f"saved checkpoint {save_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='x', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig('loss_curves.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_f1_scores) + 1), train_f1_scores, marker='o',
             label='Training F1 Score')
    plt.plot(range(1, len(val_f1_scores) + 1), val_f1_scores, marker='x',
             label='Validation F1 Score')
    plt.title('Training & Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('loss_curves.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(val_ap_scores) + 1), val_ap_scores, marker='x', label='Validation AP')
    plt.title('Validation Average Precision (AUCPR)')
    plt.xlabel('Epoch');
    plt.ylabel('AP');
    plt.grid(True);
    plt.legend();
    plt.tight_layout()
    plt.show()
    return model