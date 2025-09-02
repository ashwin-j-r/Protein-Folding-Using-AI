# preprocess.py
#
# This script handles the preprocessing of gzipped protein structure files (.cif.gz).
# It's updated to work with the AlphaFold database file structure where files
# like 'AF-O52908-F1-model_v4.cif.gz' are located in a single directory.
#
# The script performs the following actions:
# 1. Scans the dataset directory for files ending in '.cif.gz'.
# 2. Opens the gzipped files and parses them.
# 3. Extracts the raw amino acid sequence and saves it to a .txt file in the 'sequences' directory.
# 4. Extracts the 3D coordinates of alpha-carbons and saves them as a .pt file in the 'structures' directory.
# 5. Creates a combined graph object for training and saves it to the 'processed_graphs' directory.
#
# To run this script:
# 1. Make sure you have the required libraries installed:
#    pip install torch biopython tqdm torch_geometric
# 2. Create a directory named 'dataset' and place your .cif.gz files inside it.
# 3. Run the script from your terminal: python preprocess.py

import os
import torch
from torch_geometric.data import Data
from Bio.PDB import MMCIFParser
import warnings
from tqdm import tqdm
import gzip

# Suppress Biopython warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# --- Configuration ---
class Config:
    """Stores configuration variables for the project."""
    DATASET_DIR = "dataset"
    SEQUENCE_DIR = "sequences"
    STRUCTURE_DIR = "structures"
    PROCESSED_GRAPH_DIR = "processed_graphs"
    K_NEIGHBORS = 10  # Number of sequence neighbors to connect in the graph


# --- Amino Acid Vocabulary ---
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
# A more robust 3-to-1 mapping
RESIDUE_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def create_dummy_cif_file():
    """Creates a sample gzipped .cif file for demonstration purposes."""
    if not os.path.exists(Config.DATASET_DIR):
        os.makedirs(Config.DATASET_DIR)

    cif_content = """
data_AF-A0A0A0MSX3-F1
#
_entry.id   AF-A0A0A0MSX3-F1
#
_entity_poly.pdbx_seq_one_letter_code
;MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLC
VFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLV
REIRQHKLRKLNPPDESGGCMS
;
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM   1    N   N    MET   A   1    1     -2.479    -14.495   -1.564    95.21
ATOM   2    C   CA   MET   A   1    1     -1.129    -14.073   -1.215    95.21
ATOM   3    C   C    MET   A   1    1     -0.158    -15.197   -1.568    95.21
ATOM   4    O   O    MET   A   1    1     -0.455    -16.216   -1.034    95.21
ATOM   5    N   N    THR   A   1    2     1.101     -15.019   -2.021    95.21
ATOM   6    C   CA   THR   A   1    2     2.155     -15.964   -2.319    95.21
ATOM   7    C   C    THR   A   1    2     2.001     -17.222   -1.425    95.21
ATOM   8    O   O    THR   A   1    2     0.946     -17.811   -1.391    95.21
ATOM   9    N   N    GLU   A   1    3     2.999     -17.653   -0.741    95.21
ATOM   10   C   CA   GLU   A   1    3     2.969     -18.892   -0.013    95.21
"""
    # Write to a gzipped file
    dummy_filepath = os.path.join(Config.DATASET_DIR, "AF-O52908-F1-model_v4.cif.gz")
    with gzip.open(dummy_filepath, 'wt') as f:
        f.write(cif_content)
    print(f"Dummy gzipped CIF file created at '{dummy_filepath}'.")


def preprocess_cif_files():
    """
    Parses all gzipped .cif files from the dataset directory, creates separate
    sequence and structure files, and also saves combined graph objects for training.
    """
    # Create output directories if they don't exist
    for dir_path in [Config.SEQUENCE_DIR, Config.STRUCTURE_DIR, Config.PROCESSED_GRAPH_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    parser = MMCIFParser()

    # Find all gzipped CIF files in the dataset directory
    cif_files = [f for f in os.listdir(Config.DATASET_DIR) if f.endswith('.cif.gz')]

    print(f"Found {len(cif_files)} gzipped CIF files. Starting preprocessing...")
    for filename in tqdm(cif_files, desc="Preprocessing CIFs"):
        try:
            # Correctly extract the base name for saving files
            structure_id = filename.replace('.cif.gz', '')
            filepath = os.path.join(Config.DATASET_DIR, filename)

            # Open the gzipped file for parsing
            with gzip.open(filepath, 'rt') as f:
                structure = parser.get_structure(structure_id, f)

            model = structure[0]
            chain = next(model.get_chains())

            # Filter for standard amino acid residues that have an alpha-carbon
            residues = [res for res in chain.get_residues() if res.get_resname() in RESIDUE_MAP and 'CA' in res]

            if not residues:
                print(f"Warning: No valid residues found in {filename}. Skipping.")
                continue

            # 1. Extract sequence and create one-hot encoded node features
            seq = "".join([RESIDUE_MAP[res.get_resname()] for res in residues])
            node_features = []
            for aa in seq:
                feature = torch.zeros(len(AMINO_ACIDS))
                if aa in aa_to_idx:
                    feature[aa_to_idx[aa]] = 1.0
                node_features.append(feature)

            x = torch.stack(node_features)
            num_nodes = len(seq)

            # 2. Extract alpha-carbon coordinates
            ca_coords = [torch.tensor(res['CA'].get_coord(), dtype=torch.float) for res in residues]
            y = torch.stack(ca_coords)

            # 3. Create edges based on sequence proximity
            edge_indices = []
            for i in range(num_nodes):
                for j in range(max(0, i - Config.K_NEIGHBORS), min(num_nodes, i + Config.K_NEIGHBORS + 1)):
                    if i != j:
                        edge_indices.append([i, j])
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

            # --- Save the outputs to their respective directories ---

            # Save raw sequence to a text file
            with open(os.path.join(Config.SEQUENCE_DIR, f"{structure_id}_seq.txt"), "w") as f:
                f.write(seq)

            # Save coordinates tensor
            torch.save(y, os.path.join(Config.STRUCTURE_DIR, f"{structure_id}_coords.pt"))

            # Save the combined graph data object for training
            data = Data(x=x, edge_index=edge_index, y=y)
            torch.save(data, os.path.join(Config.PROCESSED_GRAPH_DIR, f"{structure_id}.pt"))

        except Exception as e:
            print(f"Could not process {filename}: {e}")


if __name__ == "__main__":
    print("--- Running Preprocessing ---")
    create_dummy_cif_file()
    preprocess_cif_files()
    print("\n--- Preprocessing Complete ---")
    print(f"Raw sequences saved in: '{Config.SEQUENCE_DIR}/'")
    print(f"Structure coordinates saved in: '{Config.STRUCTURE_DIR}/'")
    print(f"Graph data for training saved in: '{Config.PROCESSED_GRAPH_DIR}/'")
    print("You can now run train.py.")
