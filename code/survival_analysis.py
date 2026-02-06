#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
survival_analysis.py

Tarea 1: Análisis de sobrevida (Cox penalizado + Kaplan–Meier + boxplots)
para TRAIN y TEST usando:
- Embeddings por patch: df_latents_with_clinical.pkl  (1536 dims + image_path/case/label)
- Attention weights por patch: attention_weights_train.pkl, attention_weights_val.pkl
- Clínica por paciente: filtered_clinical_data.csv (OS months + OS status)

Outputs (archivos):
- km_train.png, km_test.png
- box_train.png, box_test.png
- survival_metrics.txt
- df_case_embeddings.pkl (cache de embeddings por paciente, opcional)

Requisitos:
- pandas, numpy, matplotlib
- lifelines (CoxPHFitter, KaplanMeierFitter, statistics.logrank_test, utils.concordance_index)

Instalación (conda/pip):
  pip install lifelines
"""

import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================
# Configuración
# =========================
DF_LATENTS_PKL = "df_latents_with_clinical.pkl"
ATTN_TRAIN_PKL = "attention_weights_train.pkl"
ATTN_TEST_PKL = "attention_weights_val.pkl"   # val -> test según tu decisión
CLINICAL_CSV = "filtered_clinical_data.csv"

CACHE_CASE_EMB = "df_case_embeddings.pkl"  # cache (40 x 1536 aprox)
METRICS_TXT = "survival_metrics.txt"

# Figuras (se guardan; no se muestran en terminal)
KM_TRAIN_PNG = "km_train.png"
KM_TEST_PNG = "km_test.png"
BOX_TRAIN_PNG = "box_train.png"
BOX_TEST_PNG = "box_test.png"

# Horizonte para boxplot: T = min(60, max OS months del split)
T_MAX_MONTHS = 60.0

# Grilla de penalización para Cox (Elastic Net)
# l1_ratio=1.0 -> LASSO puro; 0.0 -> Ridge
L1_RATIO = 0.1  # más ridge para estabilidad; luego puedes subir a 0.5 o 1.0 si quieres
PENALIZER_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

# CV simple en train (pocos casos). Si tu train es 32, 4-fold es razonable.
N_FOLDS = 4
RANDOM_STATE = 7

# Estabilidad numérica
EPS = 1e-12


# =========================
# Helpers
# =========================
def normalize_image_path(x) -> str:
    """
    Normaliza strings tipo tuple:
      "(/home/...jpg,)" -> "/home/...jpg"
    """
    s = str(x).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if s.endswith(","):
        s = s[:-1].strip()
    s = s.strip("'").strip('"')
    return s


def os_event_from_status(status: pd.Series) -> pd.Series:
    """
    Convierte "Overall Survival Status" (ej. "0:LIVING", "1:DECEASED") a evento:
      DECEASED -> 1
      LIVING/ALIVE -> 0
    """
    s = status.astype(str)
    ev = s.str.contains("DECEASED", case=False, na=False).astype(int)
    ev.loc[s.str.contains("LIVING", case=False, na=False)] = 0
    ev.loc[s.str.contains("ALIVE", case=False, na=False)] = 0
    return ev


def standardize_train_apply(train_X: np.ndarray, test_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estandariza con media/std de train. Devuelve (Xtr, Xte, mean, std)
    """
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return (train_X - mean) / std, (test_X - mean) / std, mean, std


def kfold_indices(n: int, k: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield tr_idx, val_idx


def safe_makedirs(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# =========================
# Paso 1: construir case-level embeddings (attention pooling)
# =========================
def build_case_embeddings_attention_pooling(
    df_latents: pd.DataFrame,
    df_attn_train: pd.DataFrame,
    df_attn_test: pd.DataFrame,
    clinical: pd.DataFrame,
    cache_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Devuelve DF a nivel paciente con columnas:
      case, split, os_months, os_event, label (si existe), + 1536 features f0..f1535
    """

    # --- Clinical: mapear Patient ID -> OS ---
    clin = clinical.copy()
    clin["Patient ID"] = clin["Patient ID"].astype(str)
    clin["os_months"] = pd.to_numeric(clin.get("Overall Survival (Months)"), errors="coerce")
    if "Overall Survival Status" in clin.columns:
        clin["os_event"] = os_event_from_status(clin["Overall Survival Status"])
    else:
        raise ValueError("No encuentro columna 'Overall Survival Status' en el CSV clínico.")

    clin_small = clin[["Patient ID", "os_months", "os_event"]].drop_duplicates("Patient ID")
    clin_small = clin_small.rename(columns={"Patient ID": "case"})

    # --- Attention weights: preparar y etiquetar split ---
    att_tr = df_attn_train.copy()
    att_te = df_attn_test.copy()

    # Normalizar image_path en weights
    att_tr["image_path_norm"] = att_tr["image_path"].map(normalize_image_path)
    att_te["image_path_norm"] = att_te["image_path"].map(normalize_image_path)

    att_tr["case_id"] = att_tr["case_id"].astype(str)
    att_te["case_id"] = att_te["case_id"].astype(str)

    att_tr["split"] = "train"
    att_te["split"] = "test"

    att = pd.concat([att_tr, att_te], ignore_index=True)

    # --- Latents: preparar ---
    if not {"image_path", "case"}.issubset(df_latents.columns):
        raise ValueError("df_latents debe tener columnas 'image_path' y 'case'.")

    lat = df_latents.copy()
    lat["case"] = lat["case"].astype(str)
    lat["image_path"] = lat["image_path"].astype(str)

    # columnas embedding (int) -> las convertimos a nombres strings f0..f1535
    emb_cols = [c for c in lat.columns if isinstance(c, (int, np.integer))]
    if len(emb_cols) == 0:
        # fallback por si vinieran como strings "0","1",...
        emb_cols = [c for c in lat.columns if isinstance(c, str) and c.isdigit()]
    if len(emb_cols) == 0:
        raise ValueError("No detecté columnas embedding (0..1535).")

    emb_cols_sorted = sorted(emb_cols, key=lambda x: int(x))
    feat_names = [f"f{int(c)}" for c in emb_cols_sorted]

    # Reordenar y asegurar float32 para ahorrar RAM
    lat_emb = lat[["image_path", "case"] + emb_cols_sorted].copy()
    lat_emb[emb_cols_sorted] = lat_emb[emb_cols_sorted].astype(np.float32)

    # --- Merge patch-level: (latents) x (weights) por image_path ---
    # Usamos weights como fuente de split (train/test) y weight por patch.
    merged = lat_emb.merge(
        att[["case_id", "image_path_norm", "attention_weight", "split", "label"]],
        left_on="image_path",
        right_on="image_path_norm",
        how="inner"
    )

    # sanity check: debería ser ~3,031,315 filas
    print("Merged rows (latents x attn):", len(merged))

    # --- Asegurar consistencia case vs case_id ---
    # En general deberían coincidir; si no, lo reportamos.
    mismatch = (merged["case"] != merged["case_id"]).mean()
    if mismatch > 0:
        print(f"WARNING: {mismatch*100:.4f}% de filas tienen case != case_id. Usaré case_id como referencia del split.")

    merged["case_final"] = merged["case_id"]

    # --- Attention-weighted pooling por case_final ---
    # z_case = sum(w*z)/sum(w)
    w = merged["attention_weight"].astype(np.float32).to_numpy()
    w = np.clip(w, 0.0, None)  # por si hubiera negativos (no debería)
    w = w + EPS

    # matriz de embeddings (N x 1536)
    Z = merged[emb_cols_sorted].to_numpy(dtype=np.float32)

    # multiplicación por pesos
    Z_w = Z * w[:, None]

    # acumular por case: usamos groupby con sum, pero sin replicar todo en pandas (más control con numpy)
    case_ids = merged["case_final"].astype(str).to_numpy()

    # codificar cases a indices 0..(n_cases-1)
    unique_cases, inv = np.unique(case_ids, return_inverse=True)
    n_cases = len(unique_cases)

    # suma ponderada por case
    sum_w = np.bincount(inv, weights=w, minlength=n_cases).astype(np.float64)
    sum_zw = np.zeros((n_cases, Z.shape[1]), dtype=np.float64)

    # acumular por bloques para no explotar memoria
    # (Z_w ya existe; si RAM es justa, puedes evitar crear Z_w y multiplicar en bloques)
    for i in range(n_cases):
        # seleccionar filas del caso i
        mask = (inv == i)
        sum_zw[i, :] = Z_w[mask].sum(axis=0)

    z_case = (sum_zw / sum_w[:, None]).astype(np.float32)

    df_case = pd.DataFrame(z_case, columns=feat_names)
    df_case.insert(0, "case", unique_cases)

    # Split y label por case (de weights)
    case_split = merged.groupby("case_final")["split"].agg(lambda x: x.iloc[0]).rename("split")
    case_label = merged.groupby("case_final")["label"].agg(lambda x: x.iloc[0]).rename("label")

    df_case = df_case.merge(case_split, left_on="case", right_index=True, how="left")
    df_case = df_case.merge(case_label, left_on="case", right_index=True, how="left")

    # Unir clínica
    df_case = df_case.merge(clin_small, on="case", how="left")

    # Chequeos
    if df_case["os_months"].isna().any() or df_case["os_event"].isna().any():
        nmiss = int(df_case["os_months"].isna().sum())
        raise ValueError(f"Faltan datos de OS en {nmiss} casos tras el merge. Revisa IDs.")

    # Guardar cache si se pidió
    if cache_path:
        df_case.to_pickle(cache_path)
        print(f"Guardado cache case embeddings: {cache_path}")

    return df_case


# =========================
# Paso 2: Cox penalizado + métricas + plots
# =========================
@dataclass
class SurvivalResults:
    penalizer_best: float
    cindex_train: float
    cindex_test: float
    pval_logrank_train: float
    pval_logrank_test: float
    T_train: float
    T_test: float


def fit_cox_penalized_select_penalizer(df_train: pd.DataFrame, feature_cols: List[str]) -> Tuple[CoxPHFitter, float, List[str]]:
    """
    Selecciona penalizer por CV en train usando c-index promedio.
    Devuelve (modelo, best_penalizer, feature_cols_filtradas).
    """

    # --- Filtrar features con varianza ~0 en TRAIN ---
    X_full = df_train[feature_cols].to_numpy(dtype=np.float32)

    # Quitar columnas con NaN/inf
    bad = ~np.isfinite(X_full).all(axis=0)
    # Quitar columnas casi constantes
    std = X_full.std(axis=0)
    low_var = std < 1e-8

    keep = ~(bad | low_var)
    feature_cols_kept = [c for c, k in zip(feature_cols, keep) if k]

    if len(feature_cols_kept) < 5:
        raise ValueError(f"Tras filtrar varianza/NaN quedaron muy pocas features: {len(feature_cols_kept)}")

    X = df_train[feature_cols_kept].to_numpy(dtype=np.float32)
    T = df_train["os_months"].to_numpy(dtype=float)
    E = df_train["os_event"].to_numpy(dtype=int)

    best_pen = None
    best_score = -np.inf

    for pen in PENALIZER_GRID:
        scores = []
        for tr_idx, va_idx in kfold_indices(len(df_train), N_FOLDS, seed=RANDOM_STATE):
            Xtr, Xva = X[tr_idx], X[va_idx]
            Ttr, Tva = T[tr_idx], T[va_idx]
            Etr, Eva = E[tr_idx], E[va_idx]

            Xtr_s, Xva_s, mean, stdv = standardize_train_apply(Xtr, Xva)

            dtr = pd.DataFrame(Xtr_s, columns=feature_cols_kept)
            dtr["os_months"] = Ttr
            dtr["os_event"] = Etr

            try:
                cph = CoxPHFitter(penalizer=pen, l1_ratio=L1_RATIO)
                cph.fit(dtr, duration_col="os_months", event_col="os_event", robust=True, show_progress=False)

                dva = pd.DataFrame(Xva_s, columns=feature_cols_kept)
                risk = cph.predict_partial_hazard(dva).values.reshape(-1)
                cidx = concordance_index(Tva, -risk, Eva)
                scores.append(float(cidx))
            except Exception:
                # si no converge en este fold, score 0
                scores.append(0.0)

        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_pen = pen

    # Ajuste final en todo train con el mejor penalizer
    Xtr_s, _, mean, stdv = standardize_train_apply(X, X)
    dtr_full = pd.DataFrame(Xtr_s, columns=feature_cols_kept)
    dtr_full["os_months"] = T
    dtr_full["os_event"] = E

    cph_final = CoxPHFitter(penalizer=best_pen, l1_ratio=L1_RATIO)
    cph_final.fit(dtr_full, duration_col="os_months", event_col="os_event", robust=True, show_progress=False)

    # Guardar mean/std
    cph_final._x_mean = mean
    cph_final._x_std = stdv

    return cph_final, float(best_pen), feature_cols_kept



def predict_risk_and_survival_at_T(cph: CoxPHFitter, X: np.ndarray, feature_cols: List[str], T_months: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve:
      risk_score: partial hazard (mayor = más riesgo)
      prob_event_before_T: 1 - S(T|x)
    """
    Xs = (X - cph._x_mean) / cph._x_std
    dX = pd.DataFrame(Xs, columns=feature_cols)

    risk = cph.predict_partial_hazard(dX).values.reshape(-1)  # exp(beta x)
    # survival function hasta T
    sf = cph.predict_survival_function(dX, times=[T_months])  # shape: (1, n)
    S_T = sf.values.reshape(-1)  # n
    prob_before_T = 1.0 - S_T
    return risk, prob_before_T


from lifelines.plotting import add_at_risk_counts

def km_plot(df_split, group_col, time_col, event_col, title, out_png, pvalue, x_max=60, show_ci=True):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    km_low = KaplanMeierFitter()
    km_high = KaplanMeierFitter()

    mask_low = df_split[group_col] == 0
    mask_high = df_split[group_col] == 1

    T_low = df_split.loc[mask_low, time_col].astype(float)
    E_low = df_split.loc[mask_low, event_col].astype(int)
    T_high = df_split.loc[mask_high, time_col].astype(float)
    E_high = df_split.loc[mask_high, event_col].astype(int)

    km_low.fit(T_low, event_observed=E_low, label=f"Low risk (n={mask_low.sum()})")
    km_high.fit(T_high, event_observed=E_high, label=f"High risk (n={mask_high.sum()})")

    km_low.plot_survival_function(ax=ax, ci_show=show_ci, censor_styles={"marker": "|", "ms": 10, "mew": 2})
    km_high.plot_survival_function(ax=ax, ci_show=show_ci, censor_styles={"marker": "|", "ms": 10, "mew": 2})

    ax.set_xlabel("Months")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"{title}\nLog-rank p = {pvalue:.3e}")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.02)

    # At-risk table
    add_at_risk_counts(km_low, km_high, ax=ax)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


from scipy.stats import mannwhitneyu

def boxplot_prob(df_split: pd.DataFrame, group_col: str, prob_col: str, title: str, out_png: str, T_months: float):
    """
    Boxplot + scatter de prob_event_before_T por grupo + Mann–Whitney U p-value.
    """
    plt.figure(figsize=(6.0, 5.2))

    data_low = df_split.loc[df_split[group_col] == 0, prob_col].astype(float).values
    data_high = df_split.loc[df_split[group_col] == 1, prob_col].astype(float).values

    # p-value no paramétrico
    ptxt = ""
    if len(data_low) >= 2 and len(data_high) >= 2:
        p = mannwhitneyu(data_low, data_high, alternative="two-sided").pvalue
        ptxt = f" | MWU p={p:.2e}"

    plt.boxplot(
        [data_low, data_high],
        tick_labels=[f"Low risk (n={len(data_low)})", f"High risk (n={len(data_high)})"],
        showfliers=False
    )

    # scatter (jitter)
    rng = np.random.default_rng(RANDOM_STATE)
    x1 = 1 + rng.normal(0, 0.04, size=len(data_low))
    x2 = 2 + rng.normal(0, 0.04, size=len(data_high))
    plt.scatter(x1, data_low, alpha=0.7)
    plt.scatter(x2, data_high, alpha=0.7)

    plt.ylim(0, 1.05)
    plt.ylabel(f"P(event < {T_months:.0f} months) = 1 - S({T_months:.0f})")
    plt.title(title + ptxt)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()




def logrank_pvalue(df_split: pd.DataFrame, group_col: str, time_col: str, event_col: str) -> float:
    """
    Log-rank test between groups 0 vs 1. Returns p-value.
    """
    g0 = df_split[group_col] == 0
    g1 = df_split[group_col] == 1
    res = logrank_test(
        df_split.loc[g0, time_col].astype(float),
        df_split.loc[g1, time_col].astype(float),
        event_observed_A=df_split.loc[g0, event_col].astype(int),
        event_observed_B=df_split.loc[g1, event_col].astype(int),
    )
    return float(res.p_value)


def bootstrap_cindex_ci(times: np.ndarray, events: np.ndarray, scores: np.ndarray,
                        n_boot: int = 2000, seed: int = 7) -> Tuple[float, float, float]:
    """
    Bootstrap del c-index con percentiles 2.5% y 97.5%.
    Nota: usamos la misma convención que antes: score alto => "mejor" (mayor tiempo).
    Si tu score es risk (alto = peor), pásalo como -risk.
    """
    rng = np.random.default_rng(seed)
    n = len(times)
    if n < 5:
        # con muy pocos puntos, CI no es estable
        c = float(concordance_index(times, scores, events))
        return c, np.nan, np.nan

    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            c = concordance_index(times[idx], scores[idx], events[idx])
            boot.append(float(c))
        except Exception:
            continue

    c_hat = float(concordance_index(times, scores, events))
    if len(boot) < 50:
        return c_hat, np.nan, np.nan

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return c_hat, float(lo), float(hi)

def hr_from_risk_group(df_split: pd.DataFrame, penalizer: float = 0.1) -> Tuple[float, float, float, float]:
    """
    Cox univariado con covariable binaria risk_group (1=high).
    Penalizer > 0 estabiliza cuando hay separación/quasi-separation.
    """
    if "risk_group" not in df_split.columns:
        raise ValueError("risk_group no existe. Crea df['risk_group'] antes de llamar hr_from_risk_group().")

    d = df_split[["os_months", "os_event", "risk_group"]].copy()

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(d, duration_col="os_months", event_col="os_event", robust=True, show_progress=False)

    coef = float(cph.params_["risk_group"])
    se = float(np.sqrt(cph.variance_matrix_.loc["risk_group", "risk_group"]))

    hr = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))
    p = float(cph.summary.loc["risk_group", "p"])
    return hr, ci_low, ci_high, p



# =========================
# Main
# =========================
def main():
    # 0) Cargar datos
    print("Cargando latents DF...")
    df_latents = pd.read_pickle(DF_LATENTS_PKL)

    print("Cargando attention weights...")
    df_attn_train = pd.read_pickle(ATTN_TRAIN_PKL)
    df_attn_test = pd.read_pickle(ATTN_TEST_PKL)

    print("Cargando clínica...")
    clinical = pd.read_csv(CLINICAL_CSV)

    # 1) Construir o cargar case embeddings
    if os.path.exists(CACHE_CASE_EMB):
        print(f"Cargando cache: {CACHE_CASE_EMB}")
        df_case = pd.read_pickle(CACHE_CASE_EMB)
    else:
        print("Construyendo embeddings por paciente (attention-weighted pooling)...")
        df_case = build_case_embeddings_attention_pooling(
            df_latents=df_latents,
            df_attn_train=df_attn_train,
            df_attn_test=df_attn_test,
            clinical=clinical,
            cache_path=CACHE_CASE_EMB
        )

    # columnas features
    feature_cols = [c for c in df_case.columns if re.match(r"^f\d+$", str(c))]

    # split
    df_train = df_case[df_case["split"] == "train"].reset_index(drop=True)
    df_test = df_case[df_case["split"] == "test"].reset_index(drop=True)

    print("\n=== Casos ===")
    print("Train cases:", len(df_train))
    print("Test cases:", len(df_test))

    # 2) Ajustar Cox penalizado en train (selección penalizer por CV)
    print("\nAjustando Cox penalizado y seleccionando penalizer por CV...")
    cph, best_pen, feature_cols_used = fit_cox_penalized_select_penalizer(df_train, feature_cols)
    feature_cols = feature_cols_used  # usar las filtradas

    print("Best penalizer:", best_pen)

    # 3) Predicción risk + prob before T para train/test
    Xtr = df_train[feature_cols].to_numpy(dtype=np.float32)
    Xte = df_test[feature_cols].to_numpy(dtype=np.float32)

    # Horizonte dinámico por split: T = min(60, max OS months del split)
    T_train = float(min(T_MAX_MONTHS, df_train["os_months"].max()))
    T_test = float(min(T_MAX_MONTHS, df_test["os_months"].max()))

    risk_tr, prob_tr = predict_risk_and_survival_at_T(cph, Xtr, feature_cols, T_train)
    risk_te, prob_te = predict_risk_and_survival_at_T(cph, Xte, feature_cols, T_test)

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["risk_score"] = risk_tr
    df_test["risk_score"] = risk_te
    df_train["prob_before_T"] = prob_tr
    df_test["prob_before_T"] = prob_te

    # 4) Definir grupos por mediana de train
    thr = float(np.median(df_train["risk_score"].values))
    df_train["risk_group"] = (df_train["risk_score"] > thr).astype(int)  # 0 low, 1 high
    df_test["risk_group"] = (df_test["risk_score"] > thr).astype(int)

    # 5) C-index (train/test) usando risk_score
    # concordance_index: convención -> mayor score = mayor tiempo. En Cox mayor risk => menor tiempo, entonces usamos -risk.
    cindex_train = float(concordance_index(df_train["os_months"], -df_train["risk_score"], df_train["os_event"]))
    cindex_test = float(concordance_index(df_test["os_months"], -df_test["risk_score"], df_test["os_event"]))

    # Bootstrap CI (usa -risk como score, igual que el c-index anterior)
    c_tr, c_tr_lo, c_tr_hi = bootstrap_cindex_ci(
        df_train["os_months"].values.astype(float),
        df_train["os_event"].values.astype(int),
        (-df_train["risk_score"].values).astype(float),
        n_boot=2000,
        seed=RANDOM_STATE
    )

    c_te, c_te_lo, c_te_hi = bootstrap_cindex_ci(
        df_test["os_months"].values.astype(float),
        df_test["os_event"].values.astype(int),
        (-df_test["risk_score"].values).astype(float),
        n_boot=2000,
        seed=RANDOM_STATE
    )


    # 6) Log-rank p-values (train/test)
    p_train = logrank_pvalue(df_train, "risk_group", "os_months", "os_event")
    p_test = logrank_pvalue(df_test, "risk_group", "os_months", "os_event")

    hr_tr, hr_tr_lo, hr_tr_hi, hr_tr_p = hr_from_risk_group(df_train, penalizer=0.0)  # train puede ir sin penalización
    hr_te, hr_te_lo, hr_te_hi, hr_te_p = hr_from_risk_group(df_test,  penalizer=0.1)  # test estabilizado


    # 7) Guardar figuras (2 por split: KM y boxplot)
    km_plot(
        df_split=df_train,
        group_col="risk_group",
        time_col="os_months",
        event_col="os_event",
        title=f"Kaplan–Meier (TRAIN) | C-index={cindex_train:.3f}",
        out_png=KM_TRAIN_PNG,
        pvalue=p_train,
        x_max=60, show_ci=True
    )
    km_plot(
        df_split=df_test,
        group_col="risk_group",
        time_col="os_months",
        event_col="os_event",
        title=f"Kaplan–Meier (TEST) | C-index={cindex_test:.3f}",
        out_png=KM_TEST_PNG,
        pvalue=p_test,
        x_max=60, show_ci=False
    )

    boxplot_prob(
        df_split=df_train,
        group_col="risk_group",
        prob_col="prob_before_T",
        title=f"Predicted risk (TRAIN) | T={T_train:.0f} months",
        out_png=BOX_TRAIN_PNG,
        T_months=T_train
    )
    boxplot_prob(
        df_split=df_test,
        group_col="risk_group",
        prob_col="prob_before_T",
        title=f"Predicted risk (TEST) | T={T_test:.0f} months",
        out_png=BOX_TEST_PNG,
        T_months=T_test
    )

    # 8) Guardar métricas
    lines = []
    lines.append("=== Survival analysis results ===")
    lines.append(f"Best penalizer: {best_pen}")
    lines.append(f"L1 ratio (1.0=LASSO): {L1_RATIO}")
    lines.append("")
    lines.append(f"C-index TRAIN: {cindex_train:.4f}")
    lines.append(f"C-index TEST : {cindex_test:.4f}")
    lines.append(f"C-index TRAIN: {c_tr:.4f}  (95% CI {c_tr_lo:.3f}–{c_tr_hi:.3f})")
    lines.append(f"C-index TEST : {c_te:.4f}  (95% CI {c_te_lo:.3f}–{c_te_hi:.3f})")
    lines.append("")
    lines.append(f"Log-rank p TRAIN: {p_train:.6e}")
    lines.append(f"Log-rank p TEST : {p_test:.6e}")
    lines.append("")
    lines.append(f"T horizon TRAIN (months): {T_train:.2f}")
    lines.append(f"T horizon TEST  (months): {T_test:.2f}")
    lines.append("")
    lines.append("Saved figures:")
    lines.append(f"- {KM_TRAIN_PNG}")
    lines.append(f"- {KM_TEST_PNG}")
    lines.append(f"- {BOX_TRAIN_PNG}")
    lines.append(f"- {BOX_TEST_PNG}")
    lines.append(f"Case-embeddings cache: {CACHE_CASE_EMB}")
    lines.append("")
    lines.append(f"HR (High vs Low) TRAIN: {hr_tr:.2f} (95% CI {hr_tr_lo:.2f}–{hr_tr_hi:.2f}), p={hr_tr_p:.2e}")
    lines.append(f"HR (High vs Low) TEST : {hr_te:.2f} (95% CI {hr_te_lo:.2f}–{hr_te_hi:.2f}), p={hr_te_p:.2e}")


    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\nListo. Abre los PNG para ver las figuras.")


if __name__ == "__main__":
    main()
