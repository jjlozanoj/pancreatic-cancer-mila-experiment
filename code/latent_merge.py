import os
import pandas as pd
from tqdm import tqdm
import glob

#Find all parts in the folder
latent_files = sorted(glob.glob("/home/jjlozanoj/NAS/Data/PAAD/Latents/latents_part*.pkl"))
print("Found latent files:", latent_files)
# Load and concat
dfs = [pd.read_pickle(f) for f in latent_files]
df_latents = pd.concat(dfs, ignore_index=True)
print("✅ Combined DataFrame shape:", df_latents.shape)
print(df_latents.head())
#Add Case ID
df_latents["case_id"] = df_latents["image_path"].apply(
    lambda x: os.path.basename(x).split("_DX")[0]
)
# Print summary before merging
print(f"📊 Latents summary: {df_latents['case_id'].nunique()} distinct cases, {len(df_latents)} rows")

# Load clinical data
clinical = pd.read_csv(
    "/home/jjlozanoj/NAS/Data/PAAD/paad_tcga_gdc_clinical_data.csv",
    usecols=["Patient ID", "Overall Survival Status", "Overall Survival (Months)"]
)
clinical["label"] = clinical["Overall Survival Status"].apply(lambda x: 1 if x == "0:LIVING" else 0)
# Merge with clinical
df_merged = df_latents.merge(clinical, left_on="case_id", right_on="Patient ID")
# Print summary after merging


def classify_and_fix_path(path):
    """Check if file exists, and fix path if it's in Removed/."""
    if os.path.exists(path):
        return "valid", path

    dirname, fname = os.path.split(path)
    removed_path = os.path.join(dirname, "Removed", fname)
    if os.path.exists(removed_path):
        return "removed", removed_path

    return "missing", path  # keep original for inspection

tqdm.pandas()
# Apply classification and path fix
df_merged[["status", "image_path"]] = df_merged["image_path"].progress_apply(
    lambda p: pd.Series(classify_and_fix_path(p))
)

# Split into subsets
df_valid   = df_merged[df_merged["status"] == "valid"]
df_removed = df_merged[df_merged["status"] == "removed"]
df_missing = df_merged[df_merged["status"] == "missing"]

print(f"✅ Valid latents:   {df_valid.shape[0]} rows, {df_valid['case_id'].nunique()} cases")
print(f"🗑 Removed latents: {df_removed.shape[0]} rows, {df_removed['case_id'].nunique()} cases")
print(f"❓ Missing latents: {df_missing.shape[0]} rows, {df_missing['case_id'].nunique()} cases")

# Save them
df_valid.to_pickle("/home/jjlozanoj/Scripts/latents/valid-latents.pkl")
df_removed.to_pickle("/home/jjlozanoj/Scripts/latents/removed-latents.pkl")
df_missing.to_pickle("/home/jjlozanoj/Scripts/latents/missing-latents.pkl")

print("💾 Saved valid, removed, and missing latent files with updated paths where needed.")