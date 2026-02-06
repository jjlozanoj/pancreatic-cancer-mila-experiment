import pandas as pd
import numpy as np
import re
import os

DF_PKL_PATH = "df_latents_with_clinical.pkl"
CLINICAL_CSV_PATH = "filtered_clinical_data.csv"
ATTN_TRAIN_PKL_PATH = "attention_weights_train.pkl"
ATTN_VAL_PKL_PATH = "attention_weights_val.pkl"

# Salidas
OUT_PKL = "df_master_meta.pkl"
OUT_CSV = None  # pon "df_master_meta.csv" si lo necesitas (pero será grande)

# Tamaños de muestreo para checks
N_SAMPLE_META = 200000
N_SAMPLE_PATH_CHECK = 100000  # 100k suele ser suficiente

def extract_wsi_id_from_path(path):
    s = str(path)
    m = re.search(r"/Patches/([^/]+)/", s)
    if m:
        return m.group(1)
    fname = s.split("/")[-1]
    m2 = re.match(r"(.+?)_(?:\d+x|20x|40x)_.+?_x\d+_y\d+\.", fname)
    if m2:
        return m2.group(1)
    m3 = re.match(r"(.+?)_x\d+_y\d+\.", fname)
    if m3:
        return m3.group(1)
    return None

def normalize_image_path(x):
    """
    Normaliza strings tipo tuple:
      "(/home/...jpg,)" -> "/home/...jpg"
    y variantes con comillas.
    """
    s = str(x).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if s.endswith(","):
        s = s[:-1].strip()
    s = s.strip("'").strip('"')
    return s

# 1) Cargar fuentes
df_lat = pd.read_pickle(DF_PKL_PATH)
df_clin = pd.read_csv(CLINICAL_CSV_PATH)
df_tr = pd.read_pickle(ATTN_TRAIN_PKL_PATH)
df_va = pd.read_pickle(ATTN_VAL_PKL_PATH)

# 2) Split por case
train_cases = set(df_tr["case_id"].astype(str).unique())
val_cases = set(df_va["case_id"].astype(str).unique())

# 3) Metadata base (sin embeddings)
meta = df_lat[["image_path","case","label"]].copy()
meta["case"] = meta["case"].astype(str)
meta["wsi_id"] = meta["image_path"].map(extract_wsi_id_from_path)

meta["split"] = np.where(meta["case"].isin(train_cases), "train",
                  np.where(meta["case"].isin(val_cases), "val", "unknown"))

# 4) Chequeos de representatividad (muestreo aleatorio)
sample = meta.sample(n=min(N_SAMPLE_META, len(meta)), random_state=7)
print("=== CONTEOS (sample aleatorio) ===")
print("cases únicos (sample):", sample["case"].nunique())
print("wsi_id únicos (sample):", sample["wsi_id"].nunique())
print("split counts (sample):")
print(sample["split"].value_counts(dropna=False))

# 5) Clínica a nivel paciente
clin = df_clin.copy()
clin["Patient ID"] = clin["Patient ID"].astype(str)

clin["os_months"] = pd.to_numeric(clin.get("Overall Survival (Months)"), errors="coerce")

# evento OS: "DECEASED" => 1, "LIVING"/"ALIVE" => 0
s = clin.get("Overall Survival Status")
if s is None:
    clin["os_event"] = np.nan
else:
    s = s.astype(str)
    clin["os_event"] = s.str.contains("DECEASED", case=False, na=False).astype(int)
    clin.loc[s.str.contains("LIVING", case=False, na=False), "os_event"] = 0
    clin.loc[s.str.contains("ALIVE", case=False, na=False), "os_event"] = 0

clin["dfs_months"] = pd.to_numeric(clin.get("Disease Free (Months)"), errors="coerce")

s2 = clin.get("Disease Free Status")
if s2 is None:
    clin["dfs_event"] = np.nan
else:
    s2 = s2.astype(str)
    # Recurred/Progressed => 1, DiseaseFree => 0
    clin["dfs_event"] = s2.str.contains("Recur|Progress", case=False, na=False).astype(int)
    clin.loc[s2.str.contains("DiseaseFree", case=False, na=False), "dfs_event"] = 0

# IMPORTANTE: si dfs_months es NaN, entonces dfs_event debería ser NaN (no 0/1)
clin.loc[clin["dfs_months"].isna(), "dfs_event"] = np.nan

clin_small = clin[["Patient ID", "os_months", "os_event", "dfs_months", "dfs_event"]].drop_duplicates("Patient ID")

# 6) Unir clínica por case
meta = meta.merge(clin_small, left_on="case", right_on="Patient ID", how="left")
meta.drop(columns=["Patient ID"], inplace=True)

# 7) Chequeos de consistencia global
print("\n=== CONSISTENCIA GLOBAL ===")
print("Total patches:", len(meta))
print("Split counts (global):")
print(meta["split"].value_counts(dropna=False))

print("\nCases en train_cases:", len(train_cases))
print("Cases en val_cases:", len(val_cases))
print("Cases en meta:", meta["case"].nunique())
print("Cases en clínica:", clin_small["Patient ID"].nunique())

missing_clin_cases = sorted(set(meta["case"].unique()) - set(clin_small["Patient ID"].unique()))
print("\nCases en latents sin clínica (count):", len(missing_clin_cases))
print("Ejemplos:", missing_clin_cases[:15])

print("\nCobertura clínica en patches (nulos):")
print("os_months nulls:", int(meta["os_months"].isna().sum()))
print("os_event nulls:", int(meta["os_event"].isna().sum()))
print("dfs_months nulls:", int(meta["dfs_months"].isna().sum()))
print("dfs_event nulls:", int(meta["dfs_event"].isna().sum()))

print("\nWSI únicos (global):", meta["wsi_id"].nunique(dropna=True))
print("WSI nulos:", int(meta["wsi_id"].isna().sum()))

# 8) CHECK image_path (mejorado)
# Aquí sí normalizamos paths de weights y evaluamos "qué porcentaje de paths de meta cae en train/val"
# sin intentar intersección masiva de 3M.
print("\n=== CHECK image_path (sample robusto) ===")
meta_paths = meta["image_path"].sample(n=min(N_SAMPLE_PATH_CHECK, len(meta)), random_state=9).astype(str)

tr_paths = df_tr["image_path"].map(normalize_image_path)
va_paths = df_va["image_path"].map(normalize_image_path)

# Para memoria: tomar sets sólo con uniques (y opcionalmente truncados)
tr_set = set(tr_paths.dropna().astype(str).unique())
va_set = set(va_paths.dropna().astype(str).unique())

hit_train = sum(p in tr_set for p in meta_paths)
hit_val = sum(p in va_set for p in meta_paths)

print(f"Sample meta paths: {len(meta_paths):,}")
print(f"Train unique paths: {len(tr_set):,} | Val unique paths: {len(va_set):,}")
print(f"Hits en train: {hit_train:,} ({(hit_train/len(meta_paths))*100:.2f}%)")
print(f"Hits en val:   {hit_val:,} ({(hit_val/len(meta_paths))*100:.2f}%)")
print("Nota: idealmente hit_train + hit_val ~ 100% si el universe de patches coincide y los paths están normalizados igual.")

# 9) Guardar DF maestro
meta.to_pickle(OUT_PKL)
print(f"\nGuardado OK: {OUT_PKL}")

if OUT_CSV:
    meta.to_csv(OUT_CSV, index=False)
    print(f"Guardado OK: {OUT_CSV}")
