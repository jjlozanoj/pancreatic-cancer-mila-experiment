"""
This script was used for the TCGA 40-case experiment.
Paths may need to be adapted for different environments.
"""
import os, fnmatch
import pandas as pd
import numpy as np

# 🔧 Set this to your patches directory
images_path = "/home/jjlozanoj/NAS/Data/PAAD/Patches/"   # e.g. your mounted Google Drive or Colab folder

records = []
for root, _, files in os.walk(images_path, topdown=True):
    if "Removed" in root:
        continue
    for file in files:
        if fnmatch.fnmatch(file, "*.jpg") or fnmatch.fnmatch(file, "*.png"):
            records.append({"path": os.path.join(root, file)})

df = pd.DataFrame(records)

# Extract case_id = everything before "_DX"
df["case_id"] = df["path"].apply(lambda x: os.path.basename(x).split("_DX")[0])

# Count patches per case
case_counts = df["case_id"].value_counts().to_dict()
total_patches = len(df)
n_splits = 5
target_per_split = total_patches / n_splits

print(f"Total patches: {total_patches}")
print(f"Target per split: ~{int(target_per_split)} patches")

# Sort cases by patch count (largest first)
sorted_cases = sorted(case_counts.items(), key=lambda x: x[1], reverse=True)

# Initialize splits
splits = [[] for _ in range(n_splits)]
split_sizes = [0] * n_splits

# Greedy assignment: put each case in the split with currently smallest patch count
for case_id, count in sorted_cases:
    idx = np.argmin(split_sizes)
    splits[idx].append(case_id)
    split_sizes[idx] += count

# Save DataFrames
for i, case_subset in enumerate(splits, 1):
    df_part = df[df["case_id"].isin(case_subset)].reset_index(drop=True)
    out_path = f"patches_cases_balanced_part{i}.pkl"
    df_part.to_pickle(out_path)
    print(f"✅ Part {i}: {len(df_part)} patches from {len(case_subset)} cases → saved to {out_path}")


#python paad_split_datrafame.py
