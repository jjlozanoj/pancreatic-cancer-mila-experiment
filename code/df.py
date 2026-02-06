import pandas as pd
import os
import re

# --- Paso 1: Leer los PKL ---
pkl_files = ["latents_part1.pkl", "latents_part2.pkl", "latents_part3.pkl", "latents_part4.pkl"]
dfs = [pd.read_pickle(f) for f in pkl_files]

# Concatenar todos en un solo dataframe
df_latents = pd.concat(dfs, ignore_index=True)

# --- Paso 2: Extraer Patient ID desde la ruta ---
# Suponiendo que el nombre del paciente es uno de los directorios en el path
# Ajusta el índice [-2] según la estructura de tu path
df_latents["case"] = df_latents["image_path"].apply(lambda x: os.path.normpath(x).split(os.sep)[-2])
df_latents["case"] = df_latents["case"].str.replace(r"_DX\d+", "", regex=True)

# --- Paso 3: Leer CSV clínico ---
df_clinical = pd.read_csv("filtered_clinical_data.csv")

# Renombrar columnas
df_clinical = df_clinical.rename(columns={
    "Patient ID": "case",
    "Patient's Vital Status": "vital_status"
})

# --- Paso 4: Mapear Alive/Dead a 1/0 ---
df_clinical["label"] = df_clinical["vital_status"].map({"Alive": 1, "Dead": 0})

# --- Paso 5: Merge por Patient ID ---
df_merged = df_latents.merge(
    df_clinical[["case", "label"]],
    on="case",
    how="left"
)

print(df_merged.head())
print(df_merged.columns)

df_merged.to_pickle("df_latents_with_clinical.pkl")