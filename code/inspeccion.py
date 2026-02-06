#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspeccion.py

Diagnóstico unificado para:
- df_latents_with_clinical.pkl  (embeddings por patch + metadata)
- filtered_clinical_data.csv    (clínica: sobrevida, covariables, IDs)
- attention_weights_train.pkl   (split train)
- attention_weights_val.pkl     (split val/test)

Incluye extracción de wsi_id desde image_path y resumen para tomar decisiones.

Uso:
  python inspeccion.py

Notas:
- Este script intenta ser robusto a distintos formatos en los PKL (dict/df/series/list/ndarray).
- Por defecto calcula nulos/dtypes SOLO en metadata del DF grande (para no demorar demasiado).
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd


# =========================
# Configuración
# =========================
DF_PKL_PATH = "df_latents_with_clinical.pkl"
CLINICAL_CSV_PATH = "filtered_clinical_data.csv"
ATTN_TRAIN_PKL_PATH = "attention_weights_train.pkl"
ATTN_VAL_PKL_PATH = "attention_weights_val.pkl"

# Si True: calcula nulos/dtypes sobre TODAS las columnas del DF grande (puede tardar y ser pesado).
FULL_NULLS_ON_LATENTS = False

# Cuántos ejemplos mostrar en cada sección
N_EXAMPLES = 10

# =========================
# Utilidades
# =========================

def is_int_colname(c: Any) -> bool:
    return isinstance(c, (int, np.integer))

def is_str_numeric(s: Any) -> bool:
    return isinstance(s, str) and s.isdigit()

def safe_value_counts(series: pd.Series, topn: int = 12):
    try:
        return series.value_counts(dropna=False).head(topn)
    except Exception as e:
        return f"No se pudo contar valores: {e}"

def find_cols(cols: List[Any], patterns: List[str]) -> List[Any]:
    out = []
    for c in cols:
        name = str(c)
        for p in patterns:
            if re.search(p, name, flags=re.IGNORECASE):
                out.append(c)
                break
    return out

def header(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

# =========================
# Extracción de wsi_id desde image_path
# =========================

def extract_wsi_id_from_path(path: Any) -> Optional[str]:
    """
    Ejemplo esperado:
      /.../Patches/TCGA-2J-AABE_DX1/TCGA-2J-AABE_DX1_20x_224px_x123_y456.jpg
    Devuelve:
      TCGA-2J-AABE_DX1

    Estrategia:
      1) Buscar el segmento entre /Patches/<wsi_id>/
      2) Fallback: extraer prefijo del filename hasta antes de '_20x_' o '_x' etc.
    """
    s = str(path)

    m = re.search(r"/Patches/([^/]+)/", s)
    if m:
        return m.group(1)

    # fallback por nombre de archivo
    fname = s.split("/")[-1]
    # patrones comunes: <wsi_id>_20x_224px_x..._y...
    m2 = re.match(r"(.+?)_(?:\d+x|20x|40x)_.+?_x\d+_y\d+\.", fname)
    if m2:
        return m2.group(1)

    # fallback más simple: antes de "_x###_y###"
    m3 = re.match(r"(.+?)_x\d+_y\d+\.", fname)
    if m3:
        return m3.group(1)

    return None


# =========================
# Diagnóstico del DF de latents
# =========================

def inspect_latents_df(df: pd.DataFrame):
    header("1) df_latents_with_clinical.pkl — Diagnóstico del DF de embeddings")

    n_rows, n_cols = df.shape
    print(f"Filas (patches): {n_rows:,}")
    print(f"Columnas totales: {n_cols:,}")

    # separar columnas embedding vs metadata
    int_named = [c for c in df.columns if is_int_colname(c)]
    str_numeric = [c for c in df.columns if is_str_numeric(c)]
    embedding_cols = int_named + [c for c in str_numeric if c not in int_named]
    meta_cols = [c for c in df.columns if c not in embedding_cols]

    header("1.1) Desglose columnas: embeddings vs metadata")
    print(f"Embeddings (colname int/dígito): {len(embedding_cols):,}")
    print(f"Metadata/Clínicas (no-embedding): {len(meta_cols):,}")
    print("Ejemplo embeddings:", embedding_cols[:min(10, len(embedding_cols))])
    print("Metadata:", meta_cols)

    header("1.2) Tipos y nulos")
    target_cols = df.columns if FULL_NULLS_ON_LATENTS else meta_cols

    if len(target_cols) == 0:
        print("No hay columnas metadata para analizar nulos/tipos (solo embeddings).")
    else:
        # construir tabla sin index mixto
        dtypes_arr = df[target_cols].dtypes.astype(str).to_numpy()
        cols_arr = np.array(list(target_cols), dtype=object)
        nulls = df[target_cols].isna().sum().to_numpy()
        null_pct = (nulls / n_rows) * 100

        info = pd.DataFrame({
            "col": cols_arr,
            "dtype": dtypes_arr,
            "nulls": nulls,
            "null_%": null_pct
        }).sort_values("nulls", ascending=False).head(30)

        print(info.to_string(index=False))

    # extracción wsi_id si existe image_path
    header("1.3) Derivación de wsi_id desde image_path (si aplica)")
    if "image_path" in df.columns:
        tmp = df[["image_path"]].head(20000).copy()  # muestra grande para estimar cobertura sin iterar todo
        tmp["wsi_id"] = tmp["image_path"].map(extract_wsi_id_from_path)

        # cobertura en el sample
        nulls = int(tmp["wsi_id"].isna().sum())
        print(f"Sample (n={len(tmp):,}) — wsi_id nulos: {nulls:,} ({(nulls/len(tmp))*100:.2f}%)")
        ex = tmp["wsi_id"].dropna().astype(str).unique()[:N_EXAMPLES]
        print("Ejemplos wsi_id (sample):", list(ex))

        # conteos rápidos a nivel global SIN crear columna en todo el DF (costoso),
        # pero sí podemos estimar WSI únicos aproximados por sample
        print(f"WSI únicos (sample): {tmp['wsi_id'].nunique(dropna=True):,}")

        print("\nSugerencia: si el porcentaje de nulos en wsi_id es ~0%,")
        print("puedes crear df['wsi_id']=... y luego usarlo para unir con splits o clínica a nivel slide.")
    else:
        print("No existe columna 'image_path' en este DF. No se puede derivar wsi_id aquí.")

    # Identificación de campos clave en metadata (si hubiera)
    header("1.4) Búsqueda de columnas clave en metadata (IDs/split/sobrevida)")
    if len(meta_cols) == 0:
        print("No hay metadata para buscar IDs/split/sobrevida.")
        return

    patterns = {
        "wsi_slide": [r"\bwsi\b", r"slide", r"svs", r"whole[_-]?slide", r"image[_-]?id", r"scan"],
        "patient":   [r"patient", r"subject", r"case[_-]?id", r"submitter", r"barcode", r"bcr", r"mrn"],
        "patch":     [r"patch", r"tile", r"coord", r"\bx\b", r"\by\b", r"row", r"col", r"level"],
        "split":     [r"split", r"fold", r"\btrain\b", r"\btest\b", r"\bval\b", r"phase", r"set"],
        "surv_time": [r"os[_-]?time", r"dfs[_-]?time", r"pfs[_-]?time", r"dss[_-]?time", r"surv.*time",
                      r"follow", r"time", r"months", r"days"],
        "surv_event":[r"os[_-]?event", r"dfs[_-]?event", r"pfs[_-]?event", r"dss[_-]?event",
                      r"\bevent\b", r"status", r"dead", r"death"],
    }

    found = {k: find_cols(meta_cols, v) for k, v in patterns.items()}
    for k, v in found.items():
        print(f"- {k}: {v}")

    header("1.5) Vista rápida (head) SOLO metadata")
    preview = df[meta_cols].head(5).copy()
    for c in preview.columns:
        if preview[c].dtype == "object":
            preview[c] = preview[c].astype(str).str.slice(0, 140)
    print(preview.to_string(index=False))


# =========================
# Diagnóstico de clínica (CSV)
# =========================

def inspect_clinical_csv(path: str):
    header("2) filtered_clinical_data.csv — Diagnóstico (clínica y llaves)")

    if not os.path.exists(path):
        print(f"No existe: {path}")
        return None

    dfc = pd.read_csv(path)
    print("shape:", dfc.shape)
    cols = list(dfc.columns)
    print("columns (preview):", cols[:80], "..." if len(cols) > 80 else "")

    header("2.1) Tipos (conteo por dtype)")
    print(dfc.dtypes.astype(str).value_counts())

    header("2.2) Head (5 filas)")
    preview = dfc.head(5).copy()
    for c in preview.columns:
        if preview[c].dtype == "object":
            preview[c] = preview[c].astype(str).str.slice(0, 140)
    print(preview.to_string(index=False))

    header("2.3) Candidatos típicos (ID / tiempo / evento)")
    id_cols = find_cols(cols, [r"case", r"patient", r"submitter", r"bcr", r"barcode", r"sample", r"\bid\b", r"tcga"])
    time_cols = find_cols(cols, [r"os", r"dfs", r"pfs", r"dss", r"time", r"days", r"months", r"follow"])
    event_cols = find_cols(cols, [r"os", r"dfs", r"pfs", r"dss", r"event", r"status", r"dead", r"death"])

    print("Candidatos ID:", id_cols[:40])
    print("Candidatos tiempo:", time_cols[:40])
    print("Candidatos evento:", event_cols[:40])

    # value_counts de uno o dos IDs candidatos (para ver formato)
    header("2.4) Ejemplos y cardinalidad de IDs (si aplica)")
    for c in id_cols[:3]:
        try:
            nun = dfc[c].nunique(dropna=True)
            print(f"- {c}: únicos={nun:,} | ejemplos={dfc[c].dropna().astype(str).unique()[:N_EXAMPLES]}")
        except Exception as e:
            print(f"- {c}: error al resumir ({e})")

    return dfc


# =========================
# Diagnóstico PKL (weights)
# =========================

def load_pickle(path: str) -> Any:
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)

def collect_identifiers_from_obj(obj: Any, max_ids: int = 200000) -> Set[str]:
    """
    Intenta extraer un conjunto de IDs (strings) desde objetos comunes:
    - dict: keys son IDs (slide/case/patch) o values contienen rutas/IDs
    - DataFrame/Series: si hay col con 'image_path'/'path'/'case'/'slide', usarla
    - list/tuple/ndarray: si es lista de strings -> IDs; si es lista de dicts, buscar campos
    Devuelve set de strings (posibles IDs).
    """
    ids: Set[str] = set()

    def add_many(vals):
        nonlocal ids
        for v in vals:
            if v is None:
                continue
            s = str(v)
            if s:
                ids.add(s)
                if len(ids) >= max_ids:
                    break

    if obj is None:
        return ids

    # dict: keys como IDs
    if isinstance(obj, dict):
        # keys
        add_many(list(obj.keys())[:max_ids])
        if len(ids) >= max_ids:
            return ids
        # values: si son estructuras con path/id
        for v in list(obj.values())[: min(2000, len(obj))]:
            if isinstance(v, dict):
                for k2 in ["image_path", "path", "case", "slide", "wsi", "svs", "id"]:
                    if k2 in v:
                        add_many([v[k2]])
            elif isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                # si es lista de cosas, tomar algunas
                sample = list(v)[:1000]
                # si parecen strings
                if sample and isinstance(sample[0], str):
                    add_many(sample)
            if len(ids) >= max_ids:
                break
        return ids

    # DataFrame
    if isinstance(obj, pd.DataFrame):
        cols = list(obj.columns)
        cand = find_cols(cols, [r"image_path", r"\bpath\b", r"case", r"slide", r"wsi", r"svs", r"\bid\b"])
        if cand:
            c = cand[0]
            add_many(obj[c].dropna().astype(str).unique()[:max_ids])
        else:
            # fallback: index si tiene strings
            if obj.index.dtype == "object":
                add_many(obj.index.astype(str).unique()[:max_ids])
        return ids

    # Series
    if isinstance(obj, pd.Series):
        if obj.index.dtype == "object":
            add_many(obj.index.astype(str).unique()[:max_ids])
        # y valores si parecen strings
        vals = obj.dropna().values
        if len(vals) > 0 and isinstance(vals[0], str):
            add_many(vals[:max_ids])
        return ids

    # list/tuple/ndarray
    if isinstance(obj, (list, tuple, np.ndarray)):
        arr = list(obj)
        if not arr:
            return ids
        # lista de strings
        if isinstance(arr[0], str):
            add_many(arr[:max_ids])
            return ids
        # lista de dicts
        if isinstance(arr[0], dict):
            for d in arr[: min(2000, len(arr))]:
                for k2 in ["image_path", "path", "case", "slide", "wsi", "svs", "id"]:
                    if k2 in d:
                        add_many([d[k2]])
                if len(ids) >= max_ids:
                    break
            return ids
        return ids

    return ids


def inspect_attention_pkl(path: str, name: str) -> Tuple[Any, Set[str]]:
    header(f"3) {name} — Diagnóstico y extracción de posibles IDs")

    if not os.path.exists(path):
        print(f"No existe: {path}")
        return None, set()

    obj = load_pickle(path)
    print("type:", type(obj))

    # describir estructura
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print("dict keys (n):", len(keys))
        print("keys sample:", keys[:N_EXAMPLES])
        if keys:
            v = obj[keys[0]]
            print("value type (1st key):", type(v))
            if isinstance(v, pd.DataFrame):
                print("value df shape:", v.shape)
                print("value df columns:", list(v.columns)[:50])
                print(v.head(3).to_string(index=False))
            elif isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                try:
                    print("value len:", len(v))
                except Exception:
                    pass
                # print sample
                try:
                    vs = list(v)[:5]
                    print("value sample:", vs)
                except Exception:
                    pass

    elif isinstance(obj, pd.DataFrame):
        print("shape:", obj.shape)
        print("columns:", list(obj.columns)[:50], "..." if obj.shape[1] > 50 else "")
        print(obj.head(3).to_string(index=False))

    elif isinstance(obj, pd.Series):
        print("series len:", len(obj))
        print("index dtype:", obj.index.dtype)
        print("head:", obj.head(3).to_string())

    elif isinstance(obj, (list, tuple, np.ndarray)):
        print("len:", len(obj))
        if len(obj) > 0:
            print("first elem type:", type(list(obj)[0]))
            try:
                print("first elem:", list(obj)[0])
            except Exception:
                pass
    else:
        print("repr (trunc):", repr(obj)[:600])

    ids = collect_identifiers_from_obj(obj)
    print(f"\nIDs extraídos (aprox): {len(ids):,}")
    if ids:
        ex = list(ids)[:N_EXAMPLES]
        print("IDs sample:", ex)

        # intentamos clasificar si parecen rutas vs TCGA IDs
        n_pathlike = sum(1 for x in ex if "/" in x or x.endswith(".svs") or x.endswith(".jpg"))
        print(f"En sample: {n_pathlike}/{len(ex)} parecen paths o nombres de archivo.")

    return obj, ids


# =========================
# Conclusiones automáticas y recomendaciones
# =========================

def summarize_next_steps(df_latents: Optional[pd.DataFrame],
                         clinical_df: Optional[pd.DataFrame],
                         train_ids: Set[str],
                         val_ids: Set[str]):
    header("4) Resumen para tomar decisiones (qué unir y cómo)")

    if df_latents is None:
        print("No se pudo cargar df_latents_with_clinical.pkl, no puedo sugerir uniones.")
        return

    # Info base del DF latents
    meta_cols = [c for c in df_latents.columns if not (is_int_colname(c) or is_str_numeric(c))]
    print(f"- Latents DF: filas={len(df_latents):,}, cols={df_latents.shape[1]:,}")
    print(f"- Metadata cols: {meta_cols}")

    # Derivar wsi_id y/o patch key en un sample
    if "image_path" in df_latents.columns:
        sample = df_latents[["image_path", "case", "label"]].head(50000).copy()
        sample["wsi_id"] = sample["image_path"].map(extract_wsi_id_from_path)
        print(f"- Sample wsi_id nulos: {sample['wsi_id'].isna().sum():,} / {len(sample):,}")
        print(f"- Sample WSI únicos: {sample['wsi_id'].nunique(dropna=True):,}")
        print(f"- Sample cases únicos: {sample['case'].nunique(dropna=True):,}")
    else:
        print("- No hay image_path: no puedo derivar wsi_id.")

    # Clínica
    if clinical_df is None:
        print("\nClínica: NO disponible (no se cargó el CSV).")
    else:
        cols = list(clinical_df.columns)
        id_cols = find_cols(cols, [r"case", r"patient", r"submitter", r"barcode", r"bcr", r"tcga", r"\bid\b"])
        time_cols = find_cols(cols, [r"os", r"dfs", r"pfs", r"dss", r"time", r"days", r"months", r"follow"])
        event_cols = find_cols(cols, [r"os", r"dfs", r"pfs", r"dss", r"event", r"status", r"dead", r"death"])

        print("\nClínica: disponible.")
        print(f"- Clinical shape: {clinical_df.shape}")
        print(f"- ID candidates: {id_cols[:10]}")
        print(f"- Time candidates: {time_cols[:10]}")
        print(f"- Event candidates: {event_cols[:10]}")

        print("\nSugerencia de unión clínica:")
        print("1) Si el CSV tiene una columna con IDs tipo 'TCGA-XX-YYYY' => unir por df_latents['case'].")
        print("2) Si el CSV tiene IDs tipo 'TCGA-XX-YYYY_DX1' o similares => unir por wsi_id derivado.")
        print("3) Si no coincide, haremos una función de normalización de IDs (strip sufijos, etc.).")

    # Splits desde attention weights
    print("\nSplits desde attention weights:")
    print(f"- Train IDs extraídos: {len(train_ids):,}")
    print(f"- Val/Test IDs extraídos: {len(val_ids):,}")

    if len(train_ids) == 0 and len(val_ids) == 0:
        print("No pude extraer IDs desde los weights; quizá el PKL no contiene IDs directamente.")
        print("En ese caso hay que inspeccionar más a fondo el objeto para ver cómo mapearlo al dataset.")
        return

    # Heurística: ¿IDs parecen image_path o case o wsi_id?
    def guess_id_level(ids: Set[str]) -> str:
        if not ids:
            return "unknown"
        xs = list(ids)[:200]
        pathlike = sum(1 for x in xs if "/" in x or x.endswith(".jpg") or x.endswith(".png") or x.endswith(".svs"))
        dxlike = sum(1 for x in xs if "_DX" in x or "_DX1" in x or "_DX2" in x)
        tcga_case = sum(1 for x in xs if re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$", x) is not None)
        tcga_any = sum(1 for x in xs if x.startswith("TCGA-"))

        if pathlike > 100:
            return "patch(image_path)-like"
        if dxlike > 80:
            return "wsi_id(slide)-like"
        if tcga_case > 80:
            return "case(patient)-like"
        if tcga_any > 80:
            return "tcga-like (needs normalization)"
        return "unknown"

    train_level = guess_id_level(train_ids)
    val_level = guess_id_level(val_ids)
    print(f"- Heurística nivel Train IDs: {train_level}")
    print(f"- Heurística nivel Val/Test IDs: {val_level}")

    print("\nRecomendación práctica:")
    print("1) Si los IDs en weights son 'patch(image_path)-like': crear split por image_path.")
    print("2) Si son 'wsi_id(slide)-like': derivar wsi_id desde image_path y asignar split por wsi_id.")
    print("3) Si son 'case(patient)-like': asignar split por case.")
    print("\nSiguiente paso: una vez sepamos el nivel, construiremos un df_unificado con columnas:")
    print("  - image_path, case, wsi_id, label, split, (clínica: time/event/...) + embeddings")
    print("y a partir de eso implementamos tus 2 tareas.")


# =========================
# Main
# =========================

def main():
    # 0) Cargar DF latents
    header("0) Carga de archivos")
    df_latents = None
    if os.path.exists(DF_PKL_PATH):
        print(f"Cargando: {DF_PKL_PATH}")
        df_latents = pd.read_pickle(DF_PKL_PATH)
        print("OK. type:", type(df_latents))
        if not isinstance(df_latents, pd.DataFrame):
            print("ERROR: el PKL no contiene un DataFrame. Abortando diagnóstico de latents.")
            df_latents = None
    else:
        print(f"No existe: {DF_PKL_PATH}")

    # 1) Diagnóstico DF latents
    if df_latents is not None:
        inspect_latents_df(df_latents)

    # 2) Diagnóstico clínica CSV
    clinical_df = inspect_clinical_csv(CLINICAL_CSV_PATH)

    # 3) Diagnóstico attention weights (train/val)
    _, train_ids = inspect_attention_pkl(ATTN_TRAIN_PKL_PATH, "attention_weights_train.pkl")
    _, val_ids = inspect_attention_pkl(ATTN_VAL_PKL_PATH, "attention_weights_val.pkl")

    # 4) Resumen para decisiones
    summarize_next_steps(df_latents, clinical_df, train_ids, val_ids)

    header("FIN")
    print("Si quieres, pega aquí el output completo (o secciones 2 y 3). Con eso te digo exactamente cómo unir todo.")


if __name__ == "__main__":
    # Evitar que pandas imprima científicamente tamaños enormes
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 180)
    main()
