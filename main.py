import os
import pickle
import random

import numpy as np
import selfies as sf
import pandas as pd

import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from rdkit.Chem import DataStructs
from torch_geometric.data import Batch

from GCN import MolecularGCN, mol_to_pyg_data
from preprocess import train_df, val_df, test_df, vocab, token2idx, idx2token
from joint import ProjectionHead
from training_modules import fit_all
from eval_plot import plot_training_curves, collect_test_embeddings, retrieval_metrics
from transformer import SelfiesTransformer, generate_all_temperatures
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from validate_generation import analyze_training_set, evaluate_generated_molecules, validate_single_molecule

# --------------------------------------------------
# Config
# --------------------------------------------------
TRAIN = False
CKPT_PATH = "artifacts/best_model.pt"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

vocab_size = len(vocab)
pad_idx = token2idx["<pad>"]
bos_idx = token2idx["<bos>"]
eos_idx = token2idx["<eos>"]
max_len = 128


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class MoleculeSelfiesDataset(Dataset):
    def __init__(self, df, token2idx):
        self.df = df.reset_index(drop=True)
        self.token2idx = token2idx
        self.unk_idx = token2idx["<unk>"]
        self.bos_idx = token2idx["<bos>"]
        self.eos_idx = token2idx["<eos>"]

    def __len__(self):
        return len(self.df)

    def encode_tokens(self, tokens):
        return [self.token2idx.get(tok, self.unk_idx) for tok in tokens]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol = row["mol"]
        graph_data = mol_to_pyg_data(mol)
        token_ids = self.encode_tokens(row["selfies_tokens"])
        full_seq = [self.bos_idx] + token_ids + [self.eos_idx]

        return {
            "graph": graph_data,
            "token_ids": torch.tensor(full_seq, dtype=torch.long),
            "canonical_smiles": row["canonical_smiles"],
            "selfies": row["selfies"],
        }


# --------------------------------------------------
# Collate
# --------------------------------------------------
def make_collate_fn(pad_idx, eos_idx):
    def collate_fn(batch):
        graphs = [item["graph"] for item in batch]
        seqs = [item["token_ids"] for item in batch]

        graph_batch = Batch.from_data_list(graphs)
        padded = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)

        input_ids = padded[:, :-1]
        target_ids = padded[:, 1:]

        eos_positions = []
        for seq in seqs:
            eos_pos = (seq == eos_idx).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                pos = eos_pos[-1].item() - 1
            else:
                pos = len(seq) - 2
            eos_positions.append(max(pos, 0))

        attention_mask = (input_ids != pad_idx).long()

        return {
            "graph": graph_batch,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
            "eos_positions": torch.tensor(eos_positions, dtype=torch.long),
            "raw_sequences": padded,
            "canonical_smiles": [item["canonical_smiles"] for item in batch],
            "selfies": [item["selfies"] for item in batch],
        }

    return collate_fn


# --------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------
def save_checkpoint(path, gcn, transformer, gcn_proj, trans_proj, history=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "gcn": gcn.state_dict(),
        "transformer": transformer.state_dict(),
        "gcn_proj": gcn_proj.state_dict(),
        "trans_proj": trans_proj.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path, gcn, transformer, gcn_proj, trans_proj, device):
    checkpoint = torch.load(path, map_location=device)
    gcn.load_state_dict(checkpoint["gcn"])
    transformer.load_state_dict(checkpoint["transformer"])
    gcn_proj.load_state_dict(checkpoint["gcn_proj"])
    trans_proj.load_state_dict(checkpoint["trans_proj"])
    print(f"Loaded checkpoint from {path}")
    return checkpoint


# --------------------------------------------------
# Model setup
# --------------------------------------------------
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

gcn_hidden_dim = 128
proj_dim = 128
transformer_hidden_dim = 192

gcn = MolecularGCN(hidden_dim=gcn_hidden_dim, dropout=0.1).to(device)

transformer = SelfiesTransformer(
    vocab_size=vocab_size,
    pad_idx=pad_idx,
    bos_idx=bos_idx,
    eos_idx=eos_idx,
    d_model=transformer_hidden_dim,
    nhead=6,
    num_layers=4,
    dim_feedforward=512,
    dropout=0.1,
    max_len=max_len,
).to(device)

gcn_proj = ProjectionHead(input_dim=gcn_hidden_dim, projection_dim=proj_dim).to(device)
trans_proj = ProjectionHead(input_dim=transformer_hidden_dim, projection_dim=proj_dim).to(device)


# --------------------------------------------------
# Data loaders
# --------------------------------------------------
batch_size = 64

train_dataset = MoleculeSelfiesDataset(train_df, token2idx)
val_dataset = MoleculeSelfiesDataset(val_df, token2idx)
test_dataset = MoleculeSelfiesDataset(test_df, token2idx)

collate_fn = make_collate_fn(pad_idx, eos_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# --------------------------------------------------
# Train or load
# --------------------------------------------------
history = None

if TRAIN:
    history, best_state = fit_all(
        gcn=gcn,
        transformer=transformer,
        gcn_proj=gcn_proj,
        trans_proj=trans_proj,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        pad_idx=pad_idx,
        warmup_epochs=15,
        joint_epochs=40,
        lr=3e-4,
        weight_decay=1e-4,
        lambda_contrastive=1.0,
        gamma_pair=0.25,
    )

    plot_training_curves(history)

    if best_state is None:
        raise RuntimeError("Training finished but best_state is None.")

    gcn.load_state_dict(best_state["gcn"])
    transformer.load_state_dict(best_state["transformer"])
    gcn_proj.load_state_dict(best_state["gcn_proj"])
    trans_proj.load_state_dict(best_state["trans_proj"])
    print("Loaded best validation checkpoint from current training run.")

    save_checkpoint(
        CKPT_PATH,
        gcn=gcn,
        transformer=transformer,
        gcn_proj=gcn_proj,
        trans_proj=trans_proj,
        history=history,
    )
else:
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CKPT_PATH}. Run once with TRAIN = True."
        )

    checkpoint = load_checkpoint(
        CKPT_PATH,
        gcn=gcn,
        transformer=transformer,
        gcn_proj=gcn_proj,
        trans_proj=trans_proj,
        device=device,
    )
    history = checkpoint.get("history", None)
    plot_training_curves(history)


# --------------------------------------------------
# Generation
# --------------------------------------------------
transformer.eval()

temperature_results = generate_all_temperatures(
    transformer=transformer,
    token2idx=token2idx,
    idx2token=idx2token,
    max_len=128,
)

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/gen_results.pkl", "wb") as f:
    pickle.dump(temperature_results, f)
print("Generation results saved to artifacts/gen_results.pkl")

generated_selfies = {
    temp: item["sequences"]
    for temp, item in temperature_results.items()
}


# --------------------------------------------------
# Section 1.3.3 validation
# --------------------------------------------------
print("\nStarting molecular generation validation process (Section 1.3.3)...")

train_smiles_list = train_df["canonical_smiles"].tolist()
train_stats = analyze_training_set(train_smiles_list)

validation_report = evaluate_generated_molecules(generated_selfies, train_stats)

for temp, metrics in validation_report.items():
    print(f"\n--- Metrics for Temperature: {temp} ---")
    print(f"RDKit Validity:    {metrics['RDKit Validity']:.4f}")
    print(f"Acceptance Rate:   {metrics['Acceptance Rate']:.4f}")
    print(f"Uniqueness:        {metrics['Uniqueness']:.4f}")
    print(f"Exact Novelty:     {metrics['Exact Novelty']:.4f}")
    print(f"Structural Novel:  {metrics['Structural Novelty']:.4f}")
    print(f"Mean QED Score:    {metrics['Mean QED']:.4f}")


# --------------------------------------------------
# Retrieval evaluation
# --------------------------------------------------
gcn.eval()
gcn_proj.eval()
trans_proj.eval()

graph_emb, seq_emb = collect_test_embeddings(
    gcn=gcn,
    transformer=transformer,
    gcn_proj=gcn_proj,
    trans_proj=trans_proj,
    dataloader=test_loader,
    device=device,
)

metrics = retrieval_metrics(graph_emb, seq_emb)
print("\nTest Retrieval Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


# --------------------------------------------------
# Detailed rejection + accepted/rejected examples + nearest neighbor
# --------------------------------------------------
accepted_examples = []
rejected_examples = []
nearest_neighbor_results = []

temps_to_check = [0.8, 1.0, 1.2]
all_accepted_smiles = []
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
for temp in temps_to_check:
    samples = generated_selfies[temp]
    rejection_counts = {}

    print(f"\n--- Detailed Rejection Analysis for Temp {temp} ---")

    for s in samples:
        smiles, status = validate_single_molecule(s, train_stats)

        if status == "passed":
            all_accepted_smiles.append(smiles)

            accepted_examples.append({
                "temperature": temp,
                "selfies": s,
                "canonical_smiles": smiles
            })

            # exact novelty check
            if smiles not in train_stats["train_canonical_smiles_set"]:
                gen_mol = Chem.MolFromSmiles(smiles)

                if gen_mol is not None:
                    gen_fp = morgan_gen.GetFingerprint(gen_mol)

                    best_sim = -1.0
                    best_idx = -1

                    for i, train_fp in enumerate(train_stats["train_fps"]):
                        sim = DataStructs.TanimotoSimilarity(gen_fp, train_fp)
                        if sim > best_sim:
                            best_sim = sim
                            best_idx = i

                    nearest_smiles = train_stats["train_canonical_smiles_list"][best_idx]

                    nearest_neighbor_results.append({
                        "temperature": temp,
                        "generated_smiles": smiles,
                        "nearest_train_smiles": nearest_smiles,
                        "max_similarity": round(best_sim, 4),
                        "similarity_distance": round(1.0 - best_sim, 4),
                        "is_structurally_novel": best_sim < 0.85
                    })

        else:
            try:
                decoded_smiles = sf.decoder(s)
            except:
                decoded_smiles = None

            rejected_examples.append({
                "temperature": temp,
                "selfies": s,
                "decoded_smiles": decoded_smiles,
                "status": status
            })

        rejection_counts[status] = rejection_counts.get(status, 0) + 1

    for status_name, count in sorted(rejection_counts.items()):
        print(f"Status: {status_name:<20} | Count: {count}")


# --------------------------------------------------
# Save CSVs
# --------------------------------------------------
pd.DataFrame(accepted_examples).to_csv("artifacts/accepted_samples.csv", index=False)
pd.DataFrame(rejected_examples).to_csv("artifacts/rejected_samples.csv", index=False)
pd.DataFrame(nearest_neighbor_results).to_csv("artifacts/nearest_neighbor_analysis.csv", index=False)

print("\n[✔] Saved: artifacts/accepted_samples.csv")
print("[✔] Saved: artifacts/rejected_samples.csv")
print("[✔] Saved: artifacts/nearest_neighbor_analysis.csv")


# --------------------------------------------------
# Accepted molecules grid
# --------------------------------------------------
all_accepted_smiles = [ex["canonical_smiles"] for ex in accepted_examples]

if len(all_accepted_smiles) >= 20:
    mols = []
    legends = []

    for i, smi in enumerate(all_accepted_smiles[:20]):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            legends.append(f"Sample {i+1}")

    if len(mols) > 0:
        grid_img = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(300, 300),
            legends=legends
        )
        grid_img.save("artifacts/accepted_molecules_grid.png")
        print("[✔] Saved: artifacts/accepted_molecules_grid.png")
else:
    print(f"[!] Only {len(all_accepted_smiles)} accepted molecules found.")


# --------------------------------------------------
# Nearest-neighbor pairs image
# --------------------------------------------------
if len(nearest_neighbor_results) >= 5:
    pair_mols = []
    pair_legends = []

    for row in nearest_neighbor_results[:5]:
        gen_mol = Chem.MolFromSmiles(row["generated_smiles"])
        nn_mol = Chem.MolFromSmiles(row["nearest_train_smiles"])

        if gen_mol is not None and nn_mol is not None:
            pair_mols.extend([gen_mol, nn_mol])
            pair_legends.extend([
                "Generated",
                f"Nearest Train\nSim={row['max_similarity']}"
            ])

    if len(pair_mols) > 0:
        nn_grid = Draw.MolsToGridImage(
            pair_mols,
            molsPerRow=2,
            subImgSize=(300, 300),
            legends=pair_legends
        )
        nn_grid.save("artifacts/nearest_neighbor_pairs.png")
        print("[✔] Saved: artifacts/nearest_neighbor_pairs.png")


# --------------------------------------------------
# Length distribution
# --------------------------------------------------
lengths = [len(list(sf.split_selfies(s))) for s in train_df["selfies"]]
plt.figure(figsize=(8, 5))
plt.hist(lengths, bins=30, color="teal", edgecolor="black", alpha=0.7)
plt.title("SELFIES Sequence Length Distribution (Train Set)")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("artifacts/length_distribution.png", dpi=300)
plt.close()
print("[✔] Saved: artifacts/length_distribution.png")