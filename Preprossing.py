# preprocess.py
#
# Updated script for preprocessing AlphaFold CIF files.
# Adds both sequence edges and spatial edges for better folding representation.
#
# Requirements:
#   pip install torch biopython tqdm torch_geometric
#
# Usage:
#   1. Place your .cif.gz files in "dataset/"
#   2. Run: python preprocess.py
#

import os
import torch
from torch_geometric.data import Data
from Bio.PDB import MMCIFParser
import warnings
from tqdm import tqdm
import gzip

# Suppress Biopython warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# --- Configuration ---
class Config:
    DATASET_DIR = "dataset"
    SEQUENCE_DIR = "sequences"
    STRUCTURE_DIR = "structures"
    PROCESSED_GRAPH_DIR = "processed_graphs"
    K_NEIGHBORS = 10      # sequence neighbors
    DIST_THRESHOLD = 8.0  # Å cutoff for spatial edges


# --- Amino Acid Vocabulary ---
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

RESIDUE_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def preprocess_cif_files():
    """
    Parses all gzipped .cif files, creates:
      - sequence txt files
      - coordinate pt files
      - graph objects with both sequence + spatial edges
    """
    # Create output dirs
    for dir_path in [Config.SEQUENCE_DIR, Config.STRUCTURE_DIR, Config.PROCESSED_GRAPH_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    parser = MMCIFParser(QUIET=True)

    # Find CIF files
    cif_files = [f for f in os.listdir(Config.DATASET_DIR) if f.endswith('.cif.gz')]
    print(f"Found {len(cif_files)} gzipped CIF files. Starting preprocessing...")

    for filename in tqdm(cif_files, desc="Preprocessing CIFs"):
        try:
            structure_id = filename.replace('.cif.gz', '')
            filepath = os.path.join(Config.DATASET_DIR, filename)

            with gzip.open(filepath, 'rt') as f:
                structure = parser.get_structure(structure_id, f)

            model = structure[0]
            chain = next(model.get_chains())

            # Keep valid residues (standard aa + CA atom present)
            residues = [res for res in chain.get_residues()
                        if res.get_resname() in RESIDUE_MAP and 'CA' in res]
            if not residues:
                print(f"⚠️ No valid residues in {filename}. Skipping.")
                continue

            # 1. Extract sequence + features
            seq = "".join([RESIDUE_MAP[res.get_resname()] for res in residues])
            node_features = []
            for aa in seq:
                feat = torch.zeros(len(AMINO_ACIDS))
                if aa in aa_to_idx:
                    feat[aa_to_idx[aa]] = 1.0
                node_features.append(feat)
            x = torch.stack(node_features)
            num_nodes = len(seq)

            # 2. Extract CA coordinates
            ca_coords = [torch.tensor(res['CA'].get_coord(), dtype=torch.float) for res in residues]
            y = torch.stack(ca_coords)

            # 3. Build edges
            edge_indices = []

            # Sequence edges (±K)
            for i in range(num_nodes):
                for j in range(max(0, i - Config.K_NEIGHBORS),
                               min(num_nodes, i + Config.K_NEIGHBORS + 1)):
                    if i != j:
                        edge_indices.append([i, j])

            # Spatial edges (distance cutoff)
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    dist = torch.norm(y[i] - y[j]).item()
                    if dist < Config.DIST_THRESHOLD:
                        edge_indices.append([i, j])
                        edge_indices.append([j, i])  # undirected

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

            # --- Save outputs ---
            with open(os.path.join(Config.SEQUENCE_DIR, f"{structure_id}_seq.txt"), "w") as f:
                f.write(seq)

            torch.save(y, os.path.join(Config.STRUCTURE_DIR, f"{structure_id}_coords.pt"))

            data = Data(x=x, edge_index=edge_index, y=y)
            torch.save(data, os.path.join(Config.PROCESSED_GRAPH_DIR, f"{structure_id}.pt"))

        except Exception as e:
            print(f"❌ Could not process {filename}: {e}")


if __name__ == "__main__":
    print("--- Running Preprocessing ---")
    preprocess_cif_files()
    print("\n--- Preprocessing Complete ---")
    print(f"Sequences saved in: '{Config.SEQUENCE_DIR}/'")
    print(f"Structures saved in: '{Config.STRUCTURE_DIR}/'")
    print(f"Graphs saved in: '{Config.PROCESSED_GRAPH_DIR}/'")
