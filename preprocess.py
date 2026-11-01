import json
import os
import selfies as sf
from rdkit import Chem
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

# --------------------------
# 1) Reproducibility
# --------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ----------------------
# Load Data
# ----------------------
df = pd.read_csv("data/Lipophilicity.csv")
assert "smiles" in df.columns, "smiles column is a wrong name!"
original_length = len(df)

# --------------------------
# Parse, sanitize, canonicalize
# --------------------------
def parse_sanitize_canonicalize(smiles):
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        Chem.SanitizeMol(mol)

        canonicalized = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        return mol ,canonicalized

    except Exception:
        return None


invalid = 0
rows = []

for idx, row in df.iterrows():
    smiles = row["smiles"]
    result = parse_sanitize_canonicalize(smiles)

    if result is None:
        invalid += 1
        continue

    mol, canonical = result
    rows.append({
        "CMPD_CHEMBLID": row["CMPD_CHEMBLID"],
        "original_smiles": smiles,
        "canonical_smiles": canonical,
        "mol": mol,
        "exp": row["exp"]
    })

clean_df = pd.DataFrame(rows)

# --------------------------
# Remove duplicated canonical molecules
# --------------------------
before_drop = len(clean_df)
clean_df = clean_df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
duplicate_removed = before_drop - len(clean_df)

# --------------------------
# Convert to SELFIES
# --------------------------
def smiles_to_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except Exception:
        return None

clean_df["selfies"] = clean_df["canonical_smiles"].apply(smiles_to_selfies)
clean_df = clean_df.dropna(subset=["selfies"]).reset_index(drop=True)

# --------------------------
# Tokenize SELFIES and remove sequences > 100 tokens
# --------------------------
def selfies_tokenizer(selfies_str):
    return list(sf.split_selfies(selfies_str))

clean_df["selfies_tokens"] = clean_df["selfies"].apply(selfies_tokenizer)
clean_df["seq_len"] = clean_df["selfies_tokens"].apply(len)

before_len_filter = len(clean_df)
clean_df = clean_df[clean_df["seq_len"] <= 100].reset_index(drop=True)

usable_count = len(clean_df)
len_filter_removed = before_len_filter - usable_count

# --------------------------
# Fixed 80/10/10 split
# --------------------------
train_val_df, test_df = train_test_split(clean_df, test_size=0.1, shuffle=True, random_state=SEED)
# Since train_val is 90%, validation should be ~10% of total => 1/9 of train_val
train_df, val_df = train_test_split(train_val_df, test_size=1/9, shuffle=True, random_state=SEED)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# --------------------------
# Build vocabulary from Train only
# --------------------------
special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

vocab_set = set()
for tokens in train_df["selfies_tokens"]:
    vocab_set.update(tokens)

vocab = special_tokens + sorted(vocab_set - set(special_tokens))
token2idx = {tok: i for i, tok in enumerate(vocab)}
idx2token = {i: tok for i, tok in enumerate(vocab)}

vocab_size = len(vocab)

# --------------------------
# Compute statistics
# --------------------------
def count_atoms(mol):
    return mol.GetNumAtoms()

def count_undirected_bonds(mol):
    return mol.GetNumBonds()

clean_df["atom_count"] = clean_df["mol"].apply(count_atoms)
clean_df["bond_count"] = clean_df["mol"].apply(count_undirected_bonds)

avg_atom_count = clean_df["atom_count"].mean()
avg_bond_count = clean_df["bond_count"].mean()

min_len = clean_df["seq_len"].min()
avg_len = clean_df["seq_len"].mean()
max_len = clean_df["seq_len"].max()


# --------------------------
# 10) Final report
# --------------------------
removed_total = original_length - usable_count

print("========== DATA PREPROCESSING REPORT ==========")
print(f"Original molecule count: {original_length}")
print(f"Removed invalid samples: {invalid}")
print(f"Removed duplicate samples: {duplicate_removed}")
print(f"Removed by length filter (>100 tokens): {len_filter_removed}")
print(f"Total removed samples: {removed_total}")
print(f"Usable molecule count: {usable_count}")
print()
print(f"Train split size: {len(train_df)}")
print(f"Validation split size: {len(val_df)}")
print(f"Test split size: {len(test_df)}")
print()
print(f"Vocabulary size: {vocab_size}")
print(f"Average atom count: {avg_atom_count:.2f}")
print(f"Average undirected bond count: {avg_bond_count:.2f}")
print(f"SELFIES length min/avg/max: {min_len} / {avg_len:.2f} / {max_len}")

# --------------------------
# Save artifacts
# --------------------------
os.makedirs("artifacts", exist_ok=True)

# Dropping mol because it's not serializable
train_save = train_df.drop(columns = ["mol"])
val_save = val_df.drop(columns = ["mol"])
test_save = test_df.drop(columns = ["mol"])
clean_save = clean_df.drop(columns = ["mol"])

# Making jsonl files (orient="records", lines=True)
train_save.to_json("artifacts/train.jsonl", orient="records", lines=True)
val_save.to_json(os.path.join("artifacts/val.jsonl"), orient="records", lines=True)
test_save.to_json(os.path.join("artifacts/test.jsonl"), orient="records", lines=True)
clean_save.to_json("artifacts/usable_all.jsonl", orient="records", lines=True)

# Making json file
with open("artifacts/vocab.json", "w",encoding="utf-8") as f:
    json.dump({
        "token2idx": token2idx,
        "idx2token": idx2token,
        "special_tokens": special_tokens
    }, f, ensure_ascii=False, indent=2)




