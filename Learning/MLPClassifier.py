import torch
import torch.nn as nn
import numpy as np
import tqdm
import copy
import matplotlib.pyplot as plt


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

def train_fold(model, train_loader, val_loader, device, epochs, lr, threshold, patience=5, min_delta=1e-5, save_model=False, rm_ft_ds=False):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state_dict = None
    optimizer_best_state_dict = None
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
            best_model_state_dict = copy.deepcopy(model.state_dict())
            optimizer_best_state_dict = copy.deepcopy(optimizer.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if patience is not None and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        if save_model:
            if rm_ft_ds:
                save_path = "saved_model/mlp_final_checkpoint_rm_ft.pth"
            else:
                save_path = "saved_model/mlp_final_checkpoint_withoutOT_test.pth"
            torch.save({
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer_best_state_dict,
            }, save_path)
            print(f"saved checkpoint {save_path}")

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