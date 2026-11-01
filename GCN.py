import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


#------------------------
# Graph conversion for GNN to use
#------------------------

# Categorical vocab helpers

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.UNSPECIFIED:0,
    Chem.rdchem.HybridizationType.S:1,
    Chem.rdchem.HybridizationType.SP:2,
    Chem.rdchem.HybridizationType.SP2:3,
    Chem.rdchem.HybridizationType.SP3:4,
    Chem.rdchem.HybridizationType.SP3D:5,
    Chem.rdchem.HybridizationType.SP3D2:6
}


CHIRALITY_MAP = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3
}

BOND_WEIGHT_MAP = {
    Chem.rdchem.BondType.SINGLE: 1.0,
    Chem.rdchem.BondType.DOUBLE: 2.0,
    Chem.rdchem.BondType.TRIPLE: 3.0,
    Chem.rdchem.BondType.AROMATIC: 1.5
}


def safe_index(value, mapping, default=0):
    return mapping.get(value, default)

def bound_int(value, min_value, max_value):
    return max(min_value, min(int(value), max_value))

# Atom feature extraction
def atom_to_categorical_features(atom):
    """
    Returns a list of categorical ids for one atom.
    Order matters and must match the model's embedding tables.
    """
    # for embedding in model we need boundaries
    atomic_num = bound_int(atom.GetAtomicNum(),0,100)
    degree = bound_int(atom.GetDegree(), 0,10)
    formal_charge = bound_int(atom.GetFormalCharge(), -5,5)
    num_h = bound_int(atom.GetTotalNumHs(), 0, 8)
    is_aromatic = int(atom.GetIsAromatic())  # 0/1

    hybrid = safe_index(atom.GetHybridization() ,HYBRIDIZATION_MAP,0)
    chiral = safe_index(atom.GetChiralTag(),CHIRALITY_MAP,0)

    # Shift formal charge from [-5, 5] to [0, 10] (embedding can't use negative indices)
    formal_charge = formal_charge + 5

    return [
        atomic_num,
        degree,
        formal_charge,
        num_h,
        is_aromatic,
        hybrid,
        chiral
    ]

def mol_to_pyg_data(mol):
    """
    Convert an RDKit mol into a PyG Data object with:
    - x: [num_nodes, num_atom_fields] categorical ids
    - edge_index: [2, num_edges]
    - edge_weight: [num_edges]
    """
    if mol is None:
        raise ValueError("Received None molecule")

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_to_categorical_features(atom))

    #  PyTorch models operate on tensors, not Python lists.
    x = torch.tensor(node_features, dtype=torch.long)

    edge_indices = []
    edge_weights = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        w = safe_index(bond.GetBondType(),BOND_WEIGHT_MAP,1.0)

        # Undirected graph: add both directions
        edge_indices.append([i,j])
        edge_indices.append([j,i])
        edge_weights.append(w)
        edge_weights.append(w)

    if len(edge_indices) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_weight = torch.empty((0,),dtype=torch.float)
    else:
        # transpose
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    # Data(PyTorch's basic graph container) object represents one molecule
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


#------------------------
# Model
#------------------------
class GCNBlock(nn.Module):
    def __init__(self, hidden_dim, drop):
        super().__init__()
        self.conv = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        residual = x

        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.relu(x)

        return residual + x


class MolecularGCN(nn.Module):
    def __init__(self, hidden_dim=128,proj_dim=128, dropout=0.1):
        super().__init__()

        # One embedding table per categorical atom field.
        # Must match the feature order from atom_to_categorical_features().
        self.atomic_num_emb = nn.Embedding(101, hidden_dim)  # 0..100
        self.degree_emb = nn.Embedding(11, hidden_dim)  # 0..10
        self.charge_emb = nn.Embedding(11, hidden_dim)  # shifted [-5,5] -> [0,10]
        self.num_h_emb = nn.Embedding(9, hidden_dim)  # 0..8
        self.aromatic_emb = nn.Embedding(2, hidden_dim)  # 0/1
        self.hybrid_emb = nn.Embedding(len(HYBRIDIZATION_MAP), hidden_dim)
        self.chiral_emb = nn.Embedding(len(CHIRALITY_MAP), hidden_dim)

        self.gcn1 = GCNBlock(hidden_dim, drop=dropout)
        self.gcn2 = GCNBlock(hidden_dim, drop=dropout)
        self.gcn3 = GCNBlock(hidden_dim, drop=dropout)


    def embed_atoms(self, x):
        """
        x: [num_nodes, 7]
        columns:
        0 atomic_num
        1 degree
        2 formal_charge
        3 num_h
        4 is_aromatic
        5 hybridization
        6 chirality
        """
        h = (
                self.atomic_num_emb(x[:, 0]) +
                self.degree_emb(x[:, 1]) +
                self.charge_emb(x[:, 2]) +
                self.num_h_emb(x[:, 3]) +
                self.aromatic_emb(x[:, 4]) +
                self.hybrid_emb(x[:, 5]) +
                self.chiral_emb(x[:, 6])
        )
        return h

    def forward(self, data):
        # data = one molecule as a graph
        # data.x = atom features for that one molecule
        # data.edge_index = bonds for that one molecule
        # data.edge_weight = bond weights for that one molecule
        x = data.x
        # edge_index -> graph connectivity [2, num_edges]
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        # batch -> which atom belongs to which molecule
        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial atom representations from summed categorical embeddings
        h = self.embed_atoms(x)

        # Three GCN layers
        h = self.gcn1(h, edge_index, edge_weight)
        h = self.gcn2(h, edge_index, edge_weight)
        h = self.gcn3(h, edge_index, edge_weight)

        # node-level representations into graph-level representations: one vector per molecule
        g = global_mean_pool(h, batch)

        # we later do projection and normalization
        return g



