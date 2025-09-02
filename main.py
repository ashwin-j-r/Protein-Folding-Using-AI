# train_improved.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from collections import defaultdict


# --- Configuration ---
class Config:
    """Stores configuration variables for the project."""
    PROCESSED_GRAPH_DIR = "processed_graphs"
    PLOTS_DIR = "plots"
    BATCH_SIZE = 2
    LEARNING_RATE = 5e-6  # Even lower learning rate
    EPOCHS = 300
    VALIDATION_SPLIT = 0.15
    N_LAYERS = 6  # More layers for capacity
    HIDDEN_DIM = 256  # Larger hidden dimension
    DROPOUT_RATE = 0.2
    WEIGHT_DECAY = 1e-6
    EARLY_STOPPING_PATIENCE = 30
    GRADIENT_CLIP = 0.1  # Tighter gradient clipping
    WARMUP_EPOCHS = 20  # Learning rate warmup


# --- Improved EGNN Model ---

class ImprovedEGNNLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_rate=Config.DROPOUT_RATE):
        super(ImprovedEGNNLayer, self).__init__()

        # Enhanced edge MLP with more capacity
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )

        # Enhanced node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim)
        )

        # Coordinate update MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # Learnable scaling factors
        self.coord_scale = nn.Parameter(torch.tensor(0.1))
        self.feature_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, coords, edge_index):
        row, col = edge_index

        # Edge Message Passing
        rel_coords = coords[row] - coords[col]
        rel_dist = torch.norm(rel_coords, p=2, dim=1, keepdim=True) + 1e-8

        edge_input = torch.cat([h[row], h[col], rel_dist], dim=1)
        edge_attr = self.edge_mlp(edge_input)

        # Coordinate Updates
        coord_mul = self.coord_mlp(edge_attr) * (rel_coords / rel_dist)
        coord_agg = torch.zeros_like(coords)
        coord_agg.index_add_(0, row, coord_mul)

        # Scaled residual connection
        coords = coords + self.coord_scale * coord_agg

        # Feature Updates
        agg_attr = torch.zeros(h.size(0), edge_attr.size(1), device=h.device)
        agg_attr.index_add_(0, row, edge_attr)

        node_input = torch.cat([h, agg_attr], dim=1)
        h_update = self.node_mlp(node_input)

        # Scaled residual connection
        h = h + self.feature_scale * h_update

        return h, coords


class ImprovedEGNN(nn.Module):
    def __init__(self, in_dim=20, hidden_dim=Config.HIDDEN_DIM, n_layers=Config.N_LAYERS):
        super(ImprovedEGNN, self).__init__()

        # Enhanced embedding
        self.embedding_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(ImprovedEGNNLayer(hidden_dim, hidden_dim))

        # Output layers
        self.embedding_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )

        self.coord_output_layer = nn.Linear(hidden_dim, 3, bias=False)

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)  # Small initialization
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, data):
        h, edge_index = data.x, data.edge_index

        # Initialize coordinates to small random values near origin
        coords = torch.randn(h.size(0), 3, device=h.device) * 0.01

        h = self.embedding_in(h)

        for layer in self.layers:
            h, coords = layer(h, coords, edge_index)

        h = self.embedding_out(h)
        final_coord_update = self.coord_output_layer(h)

        return coords + 0.1 * final_coord_update


# --- Enhanced Training Utilities ---

class ProteinDataset(Dataset):
    def __init__(self, file_list, root_dir):
        self.file_list = file_list
        self.root_dir = root_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.root_dir, self.file_list[idx])
        data = torch.load(filepath)

        # Normalize coordinates to have zero mean and unit variance
        if hasattr(data, 'y'):
            data.y = (data.y - data.y.mean(dim=0)) / (data.y.std(dim=0) + 1e-8)

        return data


def collate_fn(batch):
    return Batch.from_data_list(batch)


def calculate_rmsd(pred_coords, true_coords):
    # Center both structures
    pred_centered = pred_coords - pred_coords.mean(dim=0, keepdim=True)
    true_centered = true_coords - true_coords.mean(dim=0, keepdim=True)

    # Calculate RMSD
    rmsd = torch.sqrt(torch.mean(torch.sum((pred_centered - true_centered) ** 2, dim=1)))
    return rmsd.item()


def get_learning_rate(epoch, warmup_epochs=Config.WARMUP_EPOCHS, base_lr=Config.LEARNING_RATE):
    """Learning rate warmup and then cosine decay"""
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (Config.EPOCHS - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    all_files = [f for f in os.listdir(Config.PROCESSED_GRAPH_DIR) if f.endswith('.pt')]
    train_files, val_files = train_test_split(all_files, test_size=Config.VALIDATION_SPLIT, random_state=42)

    train_dataset = ProteinDataset(train_files, Config.PROCESSED_GRAPH_DIR)
    val_dataset = ProteinDataset(val_files, Config.PROCESSED_GRAPH_DIR)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = ImprovedEGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Training tracking
    train_losses, val_losses = [], []
    train_rmsds, val_rmsds = [], []
    best_val_loss = float('inf')

    print("Starting training with improved model...")

    for epoch in range(Config.EPOCHS):
        # Set learning rate
        lr = get_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training
        model.train()
        epoch_train_loss, epoch_train_rmsd = 0, 0
        num_batches = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            predicted_coords = model(data)
            loss = criterion(predicted_coords, data.y)
            rmsd = calculate_rmsd(predicted_coords, data.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_rmsd += rmsd
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches
        avg_train_rmsd = epoch_train_rmsd / num_batches

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_rmsd = 0, 0
        num_val_batches = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                predicted_coords = model(data)

                loss = criterion(predicted_coords, data.y)
                rmsd = calculate_rmsd(predicted_coords, data.y)

                epoch_val_loss += loss.item()
                epoch_val_rmsd += rmsd
                num_val_batches += 1

        avg_val_loss = epoch_val_loss / num_val_batches
        avg_val_rmsd = epoch_val_rmsd / num_val_batches

        # Track metrics
        train_losses.append(avg_train_loss)
        train_rmsds.append(avg_train_rmsd)
        val_losses.append(avg_val_loss)
        val_rmsds.append(avg_val_rmsd)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}, LR: {lr:.2e}")
        print(f"  Train: Loss {avg_train_loss:.4f}, RMSD {avg_train_rmsd:.2f} Å")
        print(f"  Val:   Loss {avg_val_loss:.4f}, RMSD {avg_val_rmsd:.2f} Å")
        print("-" * 50)
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{Config.EPOCHS}, LR: {lr:.2e}")
            print(f"  Train: Loss {avg_train_loss:.4f}, RMSD {avg_train_rmsd:.2f} Å")
            print(f"  Val:   Loss {avg_val_loss:.4f}, RMSD {avg_val_rmsd:.2f} Å")
            print("-" * 50)

    print("Training completed!")


if __name__ == "__main__":
    train()