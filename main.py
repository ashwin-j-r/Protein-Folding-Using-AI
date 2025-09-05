# train_improved.py
#
# Updated training script for protein folding prediction.
# - Uses pairwise distance loss (rotation/translation invariant)
# - RMSD computed with Kabsch alignment
# - Improved learning rate for faster convergence
#

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
import numpy as np

# --- Configuration ---
class Config:
    PROCESSED_GRAPH_DIR = "processed_graphs"
    PLOTS_DIR = "plots"
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4   # higher than before
    EPOCHS = 300
    VALIDATION_SPLIT = 0.15
    N_LAYERS = 6
    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.2
    WEIGHT_DECAY = 1e-6
    EARLY_STOPPING_PATIENCE = 30
    GRADIENT_CLIP = 1.0
    WARMUP_EPOCHS = 20


# --- Improved EGNN Layer ---
class ImprovedEGNNLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_rate=Config.DROPOUT_RATE):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim)
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self.coord_scale = nn.Parameter(torch.tensor(0.1))
        self.feature_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, coords, edge_index):
        row, col = edge_index
        rel_coords = coords[row] - coords[col]
        rel_dist = torch.norm(rel_coords, p=2, dim=1, keepdim=True) + 1e-8

        edge_input = torch.cat([h[row], h[col], rel_dist], dim=1)
        edge_attr = self.edge_mlp(edge_input)

        # Coordinate updates
        coord_mul = self.coord_mlp(edge_attr) * (rel_coords / rel_dist)
        coord_agg = torch.zeros_like(coords)
        coord_agg.index_add_(0, row, coord_mul)
        coords = coords + self.coord_scale * coord_agg

        # Feature updates
        agg_attr = torch.zeros(h.size(0), edge_attr.size(1), device=h.device)
        agg_attr.index_add_(0, row, edge_attr)
        node_input = torch.cat([h, agg_attr], dim=1)
        h_update = self.node_mlp(node_input)
        h = h + self.feature_scale * h_update

        return h, coords


# --- EGNN Model ---
class ImprovedEGNN(nn.Module):
    def __init__(self, in_dim=20, hidden_dim=Config.HIDDEN_DIM, n_layers=Config.N_LAYERS):
        super().__init__()
        self.embedding_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )
        self.layers = nn.ModuleList([ImprovedEGNNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.embedding_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )
        self.coord_output_layer = nn.Linear(hidden_dim, 3, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, data):
        h, edge_index = data.x, data.edge_index
        # Initialize as straight chain
        coords = torch.arange(h.size(0), device=h.device).float().unsqueeze(1).repeat(1, 3)
        coords = coords + torch.randn_like(coords) * 0.01

        h = self.embedding_in(h)
        for layer in self.layers:
            h, coords = layer(h, coords, edge_index)

        h = self.embedding_out(h)
        final_coord_update = self.coord_output_layer(h)
        return coords + 0.1 * final_coord_update


# --- Dataset ---
class ProteinDataset(Dataset):
    def __init__(self, file_list, root_dir):
        self.file_list = file_list
        self.root_dir = root_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.root_dir, self.file_list[idx])
        data = torch.load(filepath)
        # Center coordinates (translation invariance)
        if hasattr(data, 'y'):
            data.y = data.y - data.y.mean(dim=0, keepdim=True)
        return data


def collate_fn(batch):
    return Batch.from_data_list(batch)


# --- Loss + Metrics ---
def pairwise_distance_loss(pred, true):
    pdist_pred = torch.cdist(pred, pred)
    pdist_true = torch.cdist(true, true)
    return F.mse_loss(pdist_pred, pdist_true)


def kabsch_rmsd(P, Q):
    """
    Compute RMSD between P and Q using the Kabsch algorithm.
    P, Q: [N, 3] tensors
    """
    P = P - P.mean(dim=0)
    Q = Q - Q.mean(dim=0)
    C = torch.matmul(P.T, Q)
    V, S, W = torch.svd(C)
    d = torch.det(torch.matmul(W, V.T))
    if d < 0:
        V[:, -1] *= -1
    U = torch.matmul(W, V.T)
    P_rot = torch.matmul(P, U)
    rmsd = torch.sqrt(torch.mean(torch.sum((P_rot - Q) ** 2, dim=1)))
    return rmsd.item()


def get_learning_rate(epoch, warmup_epochs=Config.WARMUP_EPOCHS, base_lr=Config.LEARNING_RATE):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (Config.EPOCHS - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


# --- Training ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_files = [f for f in os.listdir(Config.PROCESSED_GRAPH_DIR) if f.endswith('.pt')]
    train_files, val_files = train_test_split(all_files, test_size=Config.VALIDATION_SPLIT, random_state=42)

    train_dataset = ProteinDataset(train_files, Config.PROCESSED_GRAPH_DIR)
    val_dataset = ProteinDataset(val_files, Config.PROCESSED_GRAPH_DIR)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = ImprovedEGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=Config.WEIGHT_DECAY)

    best_val_loss = float('inf')
    print("Starting training with improved loss + Kabsch RMSD...")

    for epoch in range(Config.EPOCHS):
        lr = get_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- Training ---
        model.train()
        epoch_train_loss, epoch_train_rmsd, num_batches = 0, 0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred_coords = model(data)
            loss = pairwise_distance_loss(pred_coords, data.y)
            rmsd = kabsch_rmsd(pred_coords.cpu(), data.y.cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_train_rmsd += rmsd
            num_batches += 1
        avg_train_loss = epoch_train_loss / num_batches
        avg_train_rmsd = epoch_train_rmsd / num_batches

        # --- Validation ---
        model.eval()
        epoch_val_loss, epoch_val_rmsd, num_val_batches = 0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred_coords = model(data)
                loss = pairwise_distance_loss(pred_coords, data.y)
                rmsd = kabsch_rmsd(pred_coords.cpu(), data.y.cpu())
                epoch_val_loss += loss.item()
                epoch_val_rmsd += rmsd
                num_val_batches += 1
        avg_val_loss = epoch_val_loss / num_val_batches
        avg_val_rmsd = epoch_val_rmsd / num_val_batches

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch+1}/{Config.EPOCHS}, LR={lr:.2e}")
        print(f"  Train: Loss {avg_train_loss:.4f}, RMSD {avg_train_rmsd:.2f} Å")
        print(f"  Val:   Loss {avg_val_loss:.4f}, RMSD {avg_val_rmsd:.2f} Å")
        print("-" * 50)

    print("Training complete!")


if __name__ == "__main__":
    train()
