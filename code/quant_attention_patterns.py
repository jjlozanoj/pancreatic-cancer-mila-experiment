#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
quant_attention_patterns_v6.py

TAREA 2 — Quantitative patterns in high-attention regions (v6)

Cambios vs v4:
- Downsampling para outputs (thumb/heat/overlay/mask) para que no pesen tanto.
- Heatmap con mejor contraste: normaliza por p99 del heat acumulado.
- Reporte TXT final (summary_report.txt) con conclusiones cuantitativas Alive vs Dead.
- Solo procesa los 40 casos que estén en df_master_meta.pkl (ignora WSIs extra).

FIX v6 (CROPS EN BLANCO):
- OpenSlide.read_region(location, level, size) SIEMPRE recibe location en coordenadas de NIVEL 0.
- Antes se estaba pasando location en coordenadas del level -> crops fuera de tejido (blancos).
- Ahora: definimos tamaño objetivo en nivel 0 (CROP_SIZE_READ, por defecto 1024),
  lo convertimos a tamaño en CROP_LEVEL y calculamos top-left EN NIVEL 0.

UPDATE (MULTI-CROPS / PEAK-CENTERED):
- Si hay varias regiones con atención alta, se extraen varios crops (hasta N_COMPONENTS_PER_CASE).
- Para cada componente conectado, el crop se centra en el PICO (máximo heat) dentro del componente,
  no en el centro geométrico del bbox. Esto es más estable.

Ejecución:
  python quant_attention_patterns_v6.py
"""

import os, re, json, time
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import openslide
from openslide import OpenSlide
from scipy.ndimage import gaussian_filter, label as cc_label, find_objects
from scipy.stats import fisher_exact

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ======================================================================
# =============================== CONFIG ===============================
# ======================================================================

# --- INPUTS ---
MASTER_META_PKL = "df_master_meta.pkl"            # meta + clínica + split (NO embeddings)
LATENTS_PKL = "df_latents_with_clinical.pkl"      # embeddings por patch + image_path (pesado)

USE_LATENTS_PARTS = False
LATENTS_PARTS = ["latents_part1.pkl", "latents_part2.pkl", "latents_part3.pkl", "latents_part4.pkl"]

ATTN_TRAIN_PKL = "attention_weights_train.pkl"
ATTN_TEST_PKL  = "attention_weights_val.pkl"      # val -> test

SVS_DIR = "/home/facanor/Documents/NAS/jjlozanoj/Data/PAAD/Images/"
OUT_ROOT = "out_patterns"

# --- PATH NORMALIZATION (deja en False si ya tienes coverage 100%) ---
USE_PATH_REPLACE = False
PATH_REPLACE_FROM = "/home/jjlozanoj/NAS/Data"
PATH_REPLACE_TO   = "/home/rdacostav/Documents/NAS/Data"
FALLBACK_JOIN_BY_BASENAME = True

# --- OUTPUT DOWNSAMPLING ---
SAVE_DOWNSAMPLE_FACTOR = 3

# --- HEATMAP ---
MAX_SIDE = 3000
TOP_FRAC = 0.05
THR_QUANTILE = 0.99
MIN_COMPONENT_PIXELS = 250
N_COMPONENTS_PER_CASE = 3
SIGMA = 1.2
CMAP = "magma"
OVERLAY_ALPHA = 0.45

# --- CROPS (FIX) ---
CROP_LEVEL = 0
CROP_SIZE_READ = 1024     # tamaño objetivo EN NIVEL 0 (ajusta: 768, 1024, etc.)

# --- QUANTIFICATION (CLUSTERING) ---
N_CLUSTERS = 8
PCA_DIM = 32
RANDOM_STATE = 7

# --- MONTAGES ---
MONTAGE_TILE = 256
MONTAGE_COLS = 5
MONTAGE_N = 25

SKIP_IF_EXISTS = True

EPS = 1e-12


# ======================================================================
# ============================== HELPERS ===============================
# ======================================================================

def banner(title: str, char: str = "="):
    line = char * 90
    print("\n" + line); print(title); print(line)

def subbanner(title: str, char: str = "-"):
    line = char * 90
    print("\n" + title); print(line)

def kv(k: str, v):
    print(f"{k:<34}: {v}")

def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize_image_path(x) -> str:
    s = str(x).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if s.endswith(","):
        s = s[:-1].strip()
    s = s.strip("'").strip('"')
    s = s.replace("\\", "/")
    return s

def maybe_replace_prefix(p: str) -> str:
    if not USE_PATH_REPLACE:
        return p
    if str(p).startswith(PATH_REPLACE_FROM):
        return PATH_REPLACE_TO + str(p)[len(PATH_REPLACE_FROM):]
    return p

def parse_xy_from_patch_path(p: str) -> Optional[Tuple[int, int]]:
    s = str(p)
    m = re.search(r"_x(\d+)_y(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"x(\d+)[^\d]+y(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def index_wsi_files(svs_dir: str):
    exts = (".svs", ".tif", ".tiff", ".ndpi", ".scn")
    files = []
    for root, _, names in os.walk(svs_dir):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
    return files

def build_case_to_svs_map(cases: List[str], svs_dir: str):
    all_wsi = index_wsi_files(svs_dir)
    case_to_paths = {c: [] for c in cases}
    for p in all_wsi:
        bn = os.path.basename(p)
        full = p
        for c in cases:
            if c in bn or (c in full):
                case_to_paths[c].append(p)

    chosen = {}
    multi = {}
    missing = []
    for c in cases:
        hits = case_to_paths.get(c, [])
        if len(hits) == 0:
            missing.append(c)
        else:
            chosen[c] = hits[0]
            if len(hits) > 1:
                multi[c] = hits[:10]
    return chosen, missing, multi

def build_thumbnail(slide: OpenSlide, max_side: int) -> Image.Image:
    w0, h0 = slide.dimensions
    scale = max(w0, h0) / float(max_side) if max(w0, h0) > max_side else 1.0
    tw, th = int(round(w0 / scale)), int(round(h0 / scale))
    return slide.get_thumbnail((tw, th)).convert("RGB")

def bbox_thumb_to_level0(bx0, by0, bx1, by1, w0, h0, tw, th):
    sx = w0 / float(tw)
    sy = h0 / float(th)
    X0 = int(round(bx0 * sx)); Y0 = int(round(by0 * sy))
    X1 = int(round(bx1 * sx)); Y1 = int(round(by1 * sy))
    X0 = max(0, min(w0 - 1, X0))
    Y0 = max(0, min(h0 - 1, Y0))
    X1 = max(0, min(w0, X1))
    Y1 = max(0, min(h0, Y1))
    return X0, Y0, X1, Y1

def overlay_heat_on_thumb(thumb: np.ndarray, heat: np.ndarray, cmap: str, alpha: float) -> np.ndarray:
    cm = plt.get_cmap(cmap)
    heat_rgb = (cm(np.clip(heat, 0, 1))[:, :, :3] * 255).astype(np.uint8)
    out = (thumb.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return out

def save_png_downsample(arr: np.ndarray, path: str, factor: int):
    im = Image.fromarray(arr)
    if factor is None or factor <= 1:
        im.save(path)
        return
    w, h = im.size
    nw = max(1, w // factor)
    nh = max(1, h // factor)
    im2 = im.resize((nw, nh), resample=Image.BILINEAR)
    im2.save(path)

def make_montage(img_paths: List[str], out_path: str, cols: int, tile: int):
    if len(img_paths) == 0:
        return
    imgs = []
    for p in img_paths:
        try:
            im = Image.open(p).convert("RGB").resize((tile, tile))
            imgs.append(im)
        except Exception:
            continue
    if len(imgs) == 0:
        return
    rows = int(np.ceil(len(imgs) / cols))
    canvas = Image.new("RGB", (cols * tile, rows * tile), (255, 255, 255))
    for i, im in enumerate(imgs):
        r = i // cols; c = i % cols
        canvas.paste(im, (c * tile, r * tile))
    canvas.save(out_path)

def detect_embedding_cols(df: pd.DataFrame) -> List:
    emb_cols = [c for c in df.columns if isinstance(c, (int, np.integer))]
    if len(emb_cols) == 0:
        emb_cols = [c for c in df.columns if isinstance(c, str) and c.isdigit()]
    if len(emb_cols) == 0:
        pat = re.compile(r"^(feat|feature|emb|embed|z|x|f)[\-_]?\d+$", re.IGNORECASE)
        emb_cols = [c for c in df.columns if isinstance(c, str) and pat.match(c)]
        if len(emb_cols) > 0:
            def sufnum(s):
                m = re.search(r"(\d+)$", str(s))
                return int(m.group(1)) if m else 10**9
            emb_cols = sorted(emb_cols, key=sufnum)
            return emb_cols
    if len(emb_cols) == 0:
        raise ValueError("No detecté columnas de embeddings en latents.")
    emb_cols = sorted(emb_cols, key=lambda x: int(x))
    return emb_cols


# ======================================================================
# ===================== ATTENTION: build per-patch weights =============
# ======================================================================

def load_attention_df(pkl_path: str, split_name: str) -> pd.DataFrame:
    dfw = pd.read_pickle(pkl_path)
    if "image_path" not in dfw.columns or "attention_weight" not in dfw.columns:
        raise ValueError(f"{pkl_path} debe tener image_path y attention_weight.")
    dfw = dfw.copy()
    dfw["image_path"] = dfw["image_path"].map(normalize_image_path).map(maybe_replace_prefix)
    dfw["attention_weight"] = pd.to_numeric(dfw["attention_weight"], errors="coerce").astype(np.float32)
    dfw["split_w"] = split_name
    dfw = dfw.dropna(subset=["image_path", "attention_weight"])
    dfw = dfw.sort_values("attention_weight").drop_duplicates(subset=["image_path"], keep="last")
    return dfw[["image_path", "attention_weight", "split_w"]]

def attach_weights_to_meta(df_meta: pd.DataFrame, dfw: pd.DataFrame, diag_dir: str) -> pd.DataFrame:
    subbanner("A.2) Merge meta + weights")
    dfm = df_meta.copy()
    dfm["image_path"] = dfm["image_path"].map(normalize_image_path).map(maybe_replace_prefix)

    merged = dfm.merge(dfw, on="image_path", how="left")

    if "split" not in merged.columns:
        merged["split"] = np.nan
    merged["split"] = merged["split_w"].combine_first(merged["split"])
    merged = merged.drop(columns=["split_w"], errors="ignore")

    cov = float(merged["attention_weight"].notna().mean() * 100.0)
    kv("Coverage by image_path", f"{cov:.2f}%")

    if FALLBACK_JOIN_BY_BASENAME and cov < 99.0:
        subbanner("A.3) Fallback merge por basename")
        dfm2 = merged.copy()
        dfm2["__base"] = dfm2["image_path"].map(lambda p: os.path.basename(str(p)))

        dfw2 = dfw.copy()
        dfw2["__base"] = dfw2["image_path"].map(lambda p: os.path.basename(str(p)))
        dfw2 = dfw2.sort_values("attention_weight").drop_duplicates(subset=["__base"], keep="last")

        missing = dfm2["attention_weight"].isna()
        fill = dfm2.loc[missing, ["__base"]].merge(dfw2[["__base","attention_weight","split_w"]], on="__base", how="left")
        dfm2.loc[missing, "attention_weight"] = fill["attention_weight"].values
        dfm2.loc[missing, "split"] = fill["split_w"].values

        cov2 = float(dfm2["attention_weight"].notna().mean() * 100.0)
        kv("Coverage after basename fallback", f"{cov2:.2f}%")
        merged = dfm2.drop(columns=["__base"], errors="ignore")

    diag_path = os.path.join(diag_dir, "merge_attention_diag.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump({
            "coverage_percent": float(merged["attention_weight"].notna().mean() * 100.0),
            "missing_rows": int(merged["attention_weight"].isna().sum())
        }, f, indent=2)
    kv("Saved", diag_path)

    return merged


# ======================================================================
# ===================== HEATMAPS / CROPS PER CASE ======================
# ======================================================================

def heat_normalize_for_display(heat: np.ndarray) -> np.ndarray:
    """Normaliza heat para display: divide por p99 y clipea."""
    h = heat.copy()
    if np.max(h) <= 0:
        return h
    v = h[h > 0]
    if v.size == 0:
        return h
    p99 = float(np.quantile(v, 0.99))
    if p99 <= 0:
        p99 = float(np.max(h))
    h = h / (p99 + EPS)
    h = np.clip(h, 0, 1)
    return h

def process_case(df_case: pd.DataFrame, case: str, svs_path: str, case_out_dir: str) -> Dict:
    ensure_dir(case_out_dir)

    d_thumb = os.path.join(case_out_dir, "00_thumb")
    d_heat  = os.path.join(case_out_dir, "01_heat")
    d_comp  = os.path.join(case_out_dir, "02_components")
    d_crop  = os.path.join(case_out_dir, "03_crops")
    for d in [d_thumb, d_heat, d_comp, d_crop]:
        ensure_dir(d)

    marker = os.path.join(d_heat, "heat_only.png")
    if SKIP_IF_EXISTS and os.path.exists(marker):
        return {"case": case, "skipped": True}

    slide = openslide.OpenSlide(svs_path)
    w0, h0 = slide.dimensions

    thumb_img = build_thumbnail(slide, MAX_SIDE)
    thumb = np.array(thumb_img)
    tw, th = thumb_img.size

    g = df_case.dropna(subset=["attention_weight"]).copy()
    paths = g["image_path"].tolist()
    weights = g["attention_weight"].astype(np.float32).to_numpy()

    n = len(weights)
    k = max(1, int(round(n * TOP_FRAC)))
    idx = np.argsort(weights)[-k:]
    top_w = weights[idx]          # crudo
    top_p = [paths[i] for i in idx]

    heat = np.zeros((th, tw), dtype=np.float32)

    patch_px = 224
    if len(top_p) > 0:
        m = re.search(r"_(\d+)px", top_p[0])
        if m:
            patch_px = int(m.group(1))

    sx = tw / float(w0); sy = th / float(h0)
    pw_t = max(1, int(round(patch_px * sx)))
    ph_t = max(1, int(round(patch_px * sy)))

    valid = 0
    w_med = float(np.median(np.abs(top_w))) + EPS

    for pth, wval in zip(top_p, top_w):
        xy = parse_xy_from_patch_path(pth)
        if xy is None:
            continue
        x0, y0 = xy
        tx = int(round(x0 * sx)); ty = int(round(y0 * sy))
        if tx < 0 or ty < 0 or tx >= tw or ty >= th:
            continue
        x1 = min(tw, tx + pw_t); y1 = min(th, ty + ph_t)
        heat[ty:y1, tx:x1] += float(wval / w_med)
        valid += 1

    if SIGMA > 0:
        heat = gaussian_filter(heat, sigma=SIGMA)

    heat_disp = heat_normalize_for_display(heat)

    if np.any(heat_disp > 0):
        thr = float(np.quantile(heat_disp[heat_disp > 0], THR_QUANTILE))
    else:
        thr = 1.0
    mask = (heat_disp >= thr).astype(np.uint8)

    lbl, _ = cc_label(mask)
    objs = find_objects(lbl)

    # ==============================================================
    # COMPONENTES + PICO (peak) dentro de cada componente (thumbnail)
    # ==============================================================
    comps = []
    for ccid, slc in enumerate(objs, start=1):
        if slc is None:
            continue
        ys, xs = slc
        cc_mask = (lbl[slc] == ccid)
        area = int(cc_mask.sum())
        if area < MIN_COMPONENT_PIXELS:
            continue

        y0b, y1b = int(ys.start), int(ys.stop)
        x0b, x1b = int(xs.start), int(xs.stop)

        # peak dentro del componente usando heat_disp
        h_local = heat_disp[slc].copy()
        h_local[~cc_mask] = -1.0
        iy, ix = np.unravel_index(np.argmax(h_local), h_local.shape)
        peak_y = y0b + int(iy)
        peak_x = x0b + int(ix)
        peak_val = float(h_local[iy, ix])

        comps.append({
            "cc_id": ccid,
            "area_thumb_px": area,
            "bbox_thumb": [x0b, y0b, x1b, y1b],
            "peak_thumb_xy": [peak_x, peak_y],
            "peak_val": peak_val
        })

    # IMPORTANTE: ordena por peak_val (más robusto) y limita a N_COMPONENTS_PER_CASE
    comps = sorted(comps, key=lambda d: d["peak_val"], reverse=True)[:N_COMPONENTS_PER_CASE]

    # --- SAVE thumb/heat/overlay/mask con downsampling ---
    thumb_path = os.path.join(d_thumb, "thumb.png")
    save_png_downsample(thumb, thumb_path, SAVE_DOWNSAMPLE_FACTOR)

    cm = plt.get_cmap(CMAP)
    heat_rgb = (cm(np.clip(heat_disp, 0, 1))[:, :, :3] * 255).astype(np.uint8)
    heat_only_path = os.path.join(d_heat, "heat_only.png")
    save_png_downsample(heat_rgb, heat_only_path, SAVE_DOWNSAMPLE_FACTOR)

    overlay = overlay_heat_on_thumb(thumb, heat_disp, CMAP, OVERLAY_ALPHA)
    heat_overlay_path = os.path.join(d_heat, "heat_overlay.png")
    save_png_downsample(overlay, heat_overlay_path, SAVE_DOWNSAMPLE_FACTOR)

    mask_rgb = np.stack([mask * 255, mask * 255, mask * 255], axis=-1).astype(np.uint8)
    mask_path = os.path.join(d_heat, "mask_high.png")
    save_png_downsample(mask_rgb, mask_path, SAVE_DOWNSAMPLE_FACTOR)

    # --- metadata JSON ---
    comps_path = os.path.join(d_comp, "components.json")
    meta = {
        "case": case,
        "svs_path": svs_path,
        "thumb_size": [int(tw), int(th)],
        "dims_level0": [int(w0), int(h0)],
        "n_patches_case": int(len(df_case)),
        "n_patches_with_weight": int(len(g)),
        "n_top_patches": int(k),
        "n_valid_top_patches_used": int(valid),
        "thr_quantile": float(THR_QUANTILE),
        "thr_value": float(thr),
        "n_components": int(len(comps)),
        "components": comps,
        "split": str(df_case["split"].iloc[0]) if "split" in df_case.columns else "unknown",
        "os_event": int(df_case["os_event"].iloc[0]) if "os_event" in df_case.columns else None,
        "mask_high_explain": "Binary mask where heatmap >= quantile THR_QUANTILE (computed on heat_disp 0..1).",
        "note_peak_centered_crops": "Crops are centered at the peak (max heat) inside each connected component."
    }
    with open(comps_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ==================================================================
    # CROPS (MULTI): 1 crop por componente, centrado en peak, size en L0
    # ==================================================================
    crop_rows = []

    ds = float(slide.level_downsamples[CROP_LEVEL])
    size_L = int(round(CROP_SIZE_READ / ds))
    size_L = max(64, size_L)
    if size_L % 2 == 1:
        size_L += 1
    half_L = size_L // 2

    # tamaño equivalente en nivel 0 para clamps
    size_L0_equiv = int(round(size_L * ds))

    for rank, comp in enumerate(comps, start=1):
        peak_x_t, peak_y_t = comp["peak_thumb_xy"]

        # convertir peak (thumbnail) -> nivel 0
        cx0 = int(round(peak_x_t * (w0 / float(tw))))
        cy0 = int(round(peak_y_t * (h0 / float(th))))

        # top-left EN NIVEL 0 (read_region lo exige)
        wx0_0 = int(round(cx0 - half_L * ds))
        wy0_0 = int(round(cy0 - half_L * ds))

        # clamp para asegurar que la ventana entra (en nivel 0)
        max_x0 = max(0, w0 - size_L0_equiv - 1)
        max_y0 = max(0, h0 - size_L0_equiv - 1)

        wx0_0 = max(0, min(wx0_0, max_x0))
        wy0_0 = max(0, min(wy0_0, max_y0))

        region = slide.read_region((wx0_0, wy0_0), CROP_LEVEL, (size_L, size_L)).convert("RGB")

        outp = os.path.join(d_crop, f"crop_cc{rank:02d}_L{CROP_LEVEL}_L0{CROP_SIZE_READ}.png")
        region.save(outp)

        crop_rows.append({
            "case": case,
            "split": str(df_case["split"].iloc[0]) if "split" in df_case.columns else "unknown",
            "os_event": int(df_case["os_event"].iloc[0]) if "os_event" in df_case.columns else None,
            "cc_rank": rank,
            "cc_area_thumb_px": comp["area_thumb_px"],
            "cc_peak_val": float(comp["peak_val"]),
            "peak_thumb_xy": comp["peak_thumb_xy"],
            "crop_level": int(CROP_LEVEL),
            "crop_size_level0": int(CROP_SIZE_READ),
            "crop_size_level": int(size_L),
            "downsample": float(ds),
            "top_left_level0": [int(wx0_0), int(wy0_0)],
            "crop_path": outp
        })

    slide.close()

    return {
        "case": case,
        "thumb_path": thumb_path,
        "heat_only_path": heat_only_path,
        "heat_overlay_path": heat_overlay_path,
        "mask_path": mask_path,
        "components_json": comps_path,
        "crop_rows": crop_rows
    }


# ======================================================================
# ========================== QUANTIFICATION ============================
# ======================================================================

def get_top_patch_set_from_meta(df_meta_w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, g in df_meta_w.groupby("case"):
        g = g.dropna(subset=["attention_weight"])
        if len(g) == 0:
            continue
        g = g.sort_values("attention_weight")
        k = max(1, int(round(len(g) * TOP_FRAC)))
        rows.append(g.iloc[-k:][["case","image_path","attention_weight","os_event","split"]])
    return pd.concat(rows, ignore_index=True) if len(rows) else pd.DataFrame()

def load_latents_df() -> pd.DataFrame:
    if USE_LATENTS_PARTS:
        parts = [pd.read_pickle(p) for p in LATENTS_PARTS]
        return pd.concat(parts, ignore_index=True)
    return pd.read_pickle(LATENTS_PKL)

def quantify_patterns_with_latents(df_top: pd.DataFrame, quant_out_dir: str):
    ensure_dir(quant_out_dir)
    banner("C) Cuantificación: embeddings top-att + clustering")

    kv("Top patches (meta)", len(df_top))
    kv("Top cases", df_top["case"].nunique())

    subbanner("C.1) Cargando latents (embeddings)")
    lat = load_latents_df()
    if "image_path" not in lat.columns:
        raise ValueError("Latents DF debe tener image_path.")
    lat = lat.copy()
    lat["image_path"] = lat["image_path"].map(normalize_image_path).map(maybe_replace_prefix)

    emb_cols = detect_embedding_cols(lat)
    kv("Latents rows", len(lat))
    kv("Embedding dims", len(emb_cols))

    subbanner("C.2) Merge top-att patches con latents por image_path")
    top_paths = set(df_top["image_path"].tolist())
    lat_f = lat[lat["image_path"].isin(top_paths)].copy()
    kv("Latents filtered rows (match top_paths)", len(lat_f))

    merged = df_top.merge(lat_f[["image_path"] + emb_cols], on="image_path", how="inner")
    kv("Merged rows", len(merged))
    if len(merged) == 0:
        raise ValueError("No hubo intersección entre top patches y latents. Revisa normalización de paths.")

    out_embed = os.path.join(quant_out_dir, "top_patches_with_embeddings_meta.csv")
    merged[["case","split","os_event","attention_weight","image_path"]].to_csv(out_embed, index=False)
    kv("Saved", out_embed)

    subbanner("C.3) PCA + KMeans")
    X = merged[emb_cols].to_numpy(dtype=np.float32)
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    pca_dim = min(PCA_DIM, Xs.shape[1], max(2, Xs.shape[0]-1))
    Z = PCA(n_components=pca_dim, random_state=RANDOM_STATE).fit_transform(Xs)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    merged["cluster"] = km.fit_predict(Z)

    out1 = os.path.join(quant_out_dir, "top_patches_with_clusters.csv")
    merged[["case","split","os_event","attention_weight","cluster","image_path"]].to_csv(out1, index=False)
    kv("Saved", out1)

    subbanner("C.4) Proporciones por caso")
    tab = merged.groupby(["case","os_event","cluster"]).size().reset_index(name="count")
    tot = tab.groupby(["case","os_event"])["count"].sum().reset_index(name="total")
    tab = tab.merge(tot, on=["case","os_event"], how="left")
    tab["prop"] = tab["count"] / tab["total"]
    piv = tab.pivot_table(index=["case","os_event"], columns="cluster", values="prop", fill_value=0.0).reset_index()
    out2 = os.path.join(quant_out_dir, "case_cluster_props.csv")
    piv.to_csv(out2, index=False)
    kv("Saved", out2)

    subbanner("C.5) Enriquecimiento Alive vs Dead (Fisher sobre presencia)")
    results = []
    for cl in range(N_CLUSTERS):
        props = piv[cl].values
        ev = piv["os_event"].values
        pres = (props > 0).astype(int)

        a = int(((pres==1)&(ev==1)).sum())
        b = int(((pres==0)&(ev==1)).sum())
        c2= int(((pres==1)&(ev==0)).sum())
        d2= int(((pres==0)&(ev==0)).sum())

        try:
            odds, pval = fisher_exact([[a,b],[c2,d2]])
        except Exception:
            odds, pval = np.nan, np.nan

        results.append({
            "cluster": int(cl),
            "mean_prop_dead": float(props[ev==1].mean()) if np.any(ev==1) else np.nan,
            "mean_prop_alive": float(props[ev==0].mean()) if np.any(ev==0) else np.nan,
            "fisher_odds": float(odds) if np.isfinite(odds) else np.nan,
            "fisher_p": float(pval) if np.isfinite(pval) else np.nan,
            "dead_present": a, "dead_absent": b,
            "alive_present": c2, "alive_absent": d2
        })

    enrich = pd.DataFrame(results).sort_values("fisher_p", na_position="last")
    out3 = os.path.join(quant_out_dir, "cluster_enrichment.csv")
    enrich.to_csv(out3, index=False)
    kv("Saved", out3)

    vec_cols = [c for c in piv.columns if isinstance(c, int)]
    alive = piv[piv["os_event"] == 0][vec_cols].to_numpy(dtype=float)
    dead  = piv[piv["os_event"] == 1][vec_cols].to_numpy(dtype=float)
    c_alive = alive.mean(axis=0) if alive.size else np.zeros(len(vec_cols))
    c_dead  = dead.mean(axis=0) if dead.size else np.zeros(len(vec_cols))
    l1 = float(np.sum(np.abs(c_alive - c_dead)))
    l2 = float(np.linalg.norm(c_alive - c_dead))

    global_metrics = {
        "centroid_L1_distance": l1,
        "centroid_L2_distance": l2,
        "n_alive": int((piv["os_event"] == 0).sum()),
        "n_dead": int((piv["os_event"] == 1).sum())
    }
    gm_path = os.path.join(quant_out_dir, "global_group_separation.json")
    with open(gm_path, "w", encoding="utf-8") as f:
        json.dump(global_metrics, f, indent=2)
    kv("Saved", gm_path)

    return enrich, global_metrics


# ======================================================================
# ================================ MAIN ================================
# ======================================================================

def build_run_dir(out_root: str) -> str:
    ensure_dir(out_root)
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, run_id)
    ensure_dir(run_dir)
    return run_dir

def write_summary_txt(path: str, meta_w: pd.DataFrame, enrich: pd.DataFrame, global_metrics: Dict):
    lines = []
    lines.append("TAREA 2 — SUMMARY REPORT\n")
    lines.append(f"Generated at: {ts()}\n")

    lines.append("\n=== Dataset ===\n")
    lines.append(f"Cases: {meta_w['case'].nunique()}\n")
    lines.append(f"Split counts (patch-level): {meta_w['split'].value_counts(dropna=False).to_dict()}\n")
    lines.append(f"Weight coverage (%): {meta_w['attention_weight'].notna().mean()*100:.2f}\n")

    lines.append("\n=== mask_high.png ===\n")
    lines.append(f"- mask_high.png is a binary mask where heat_disp >= quantile {THR_QUANTILE}.\n")
    lines.append("- White pixels = top high-attention regions; connected components are candidate ROIs.\n")
    lines.append(f"- We extract up to {N_COMPONENTS_PER_CASE} crops per case, centered at the PEAK of each component.\n")

    lines.append("\n=== Quantitative group separation (Alive vs Dead) ===\n")
    lines.append(f"- centroid_L1_distance: {global_metrics.get('centroid_L1_distance'):.4f}\n")
    lines.append(f"- centroid_L2_distance: {global_metrics.get('centroid_L2_distance'):.4f}\n")
    lines.append(f"- n_alive: {global_metrics.get('n_alive')} | n_dead: {global_metrics.get('n_dead')}\n")

    lines.append("\n=== Cluster enrichment (Fisher) ===\n")
    if enrich is None or len(enrich) == 0:
        lines.append("No enrichment table available.\n")
    else:
        top = enrich.head(5).copy()
        lines.append("Top 5 clusters (smallest Fisher p):\n")
        lines.append(top.to_string(index=False))
        lines.append("\n\nInterpretation guide:\n")
        lines.append("- Low p-values + consistent mean_prop difference => recurring patterns differ Alive vs Dead.\n")
        lines.append("- Mostly large p-values + small centroid distances => patterns are heterogeneous/shared.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    banner("TAREA 2 — Quantitative patterns in high-attention regions (v6)")
    kv("Start", ts())
    kv("MASTER_META_PKL", MASTER_META_PKL)
    kv("LATENTS_PKL", LATENTS_PKL if not USE_LATENTS_PARTS else str(LATENTS_PARTS))
    kv("ATTN_TRAIN_PKL", ATTN_TRAIN_PKL)
    kv("ATTN_TEST_PKL", ATN_TEST_PKL if 'ATN_TEST_PKL' in globals() else ATTN_TEST_PKL)  # keep robust
    kv("SVS_DIR", SVS_DIR)
    kv("OUT_ROOT", OUT_ROOT)
    kv("SAVE_DOWNSAMPLE_FACTOR", SAVE_DOWNSAMPLE_FACTOR)
    kv("CROP_LEVEL", CROP_LEVEL)
    kv("CROP_SIZE_READ", CROP_SIZE_READ)

    run_dir = build_run_dir(OUT_ROOT)
    d_logs = os.path.join(run_dir, "00_logs")
    d_tables = os.path.join(run_dir, "01_tables")
    d_cases = os.path.join(run_dir, "02_cases")
    d_quant = os.path.join(run_dir, "03_quant")
    d_mont = os.path.join(run_dir, "04_montages")
    d_diag = os.path.join(run_dir, "05_diagnostics")
    for d in [d_logs, d_tables, d_cases, d_quant, d_mont, d_diag]:
        ensure_dir(d)
    kv("Run dir", run_dir)

    banner("A) Carga meta + weights")
    meta = pd.read_pickle(MASTER_META_PKL).copy()
    if "case" not in meta.columns or "image_path" not in meta.columns:
        raise ValueError("Meta debe tener case e image_path.")
    if "os_event" not in meta.columns:
        raise ValueError("Meta debe traer os_event (0 alive, 1 dead).")

    meta["case"] = meta["case"].astype(str)
    meta["image_path"] = meta["image_path"].map(normalize_image_path).map(maybe_replace_prefix)

    dfw_train = load_attention_df(ATTN_TRAIN_PKL, "train")
    dfw_test  = load_attention_df(ATTN_TEST_PKL,  "test")
    dfw = pd.concat([dfw_train, dfw_test], ignore_index=True)
    dfw = dfw.sort_values("attention_weight").drop_duplicates(subset=["image_path"], keep="last")

    meta_w = attach_weights_to_meta(meta, dfw, diag_dir=d_diag)

    kv("Meta rows", len(meta_w))
    kv("Cases", meta_w["case"].nunique())
    kv("Split counts", meta_w["split"].value_counts(dropna=False).to_dict())
    kv("Weight coverage %", f"{meta_w['attention_weight'].notna().mean()*100:.2f}")

    meta_w_path = os.path.join(d_tables, "meta_with_attention.pkl")
    meta_w.to_pickle(meta_w_path)
    kv("Saved", meta_w_path)

    banner("B) Heatmaps + crops por caso")
    cases = sorted(meta_w["case"].unique().tolist())
    svs_map, missing_cases, multi_cases = build_case_to_svs_map(cases, SVS_DIR)

    kv("WSI files indexed", len(index_wsi_files(SVS_DIR)))
    kv("Cases with WSI found", len(svs_map))
    kv("Missing cases", len(missing_cases))
    kv("Cases to process", len(cases))

    case_summary = []
    crop_rows_all = []
    missing_svs = []
    skipped = 0

    for i, case in enumerate(cases, start=1):
        svs = svs_map.get(case, None)
        if svs is None:
            missing_svs.append(case)
            print(f"[{i:02d}/{len(cases):02d}] {case}: SVS no encontrado -> skip")
            continue

        g = meta_w[meta_w["case"] == case].copy()
        if g["split"].nunique() > 1:
            g["split"] = g["split"].value_counts().idxmax()

        print(f"[{i:02d}/{len(cases):02d}] Procesando {case} | patches={len(g)} | split={g['split'].iloc[0]}")
        info = process_case(g, case, svs, os.path.join(d_cases, case))

        if info.get("skipped", False):
            skipped += 1
            continue

        case_summary.append({
            "case": case,
            "thumb_path": info.get("thumb_path"),
            "heat_only_path": info.get("heat_only_path"),
            "heat_overlay_path": info.get("heat_overlay_path"),
            "mask_path": info.get("mask_path"),
            "components_json": info.get("components_json")
        })
        crop_rows_all.extend(info.get("crop_rows", []))

    pd.DataFrame(case_summary).to_csv(os.path.join(d_tables, "case_heatmap_summary.csv"), index=False)
    pd.DataFrame(crop_rows_all).to_csv(os.path.join(d_tables, "case_components_and_crops.csv"), index=False)
    kv("Skipped (exists)", skipped)
    kv("Missing SVS", len(missing_svs))

    banner("C) Top-att patch set -> merge con latents -> clustering")
    df_top = get_top_patch_set_from_meta(meta_w)
    top_path = os.path.join(d_tables, "top_patches_meta.csv")
    df_top.to_csv(top_path, index=False)
    kv("Saved", top_path)

    enrich, global_metrics = quantify_patterns_with_latents(df_top, quant_out_dir=d_quant)

    banner("D) Montajes rápidos (crops rank=1 Alive vs Dead)")
    comp_df = pd.DataFrame(crop_rows_all)
    if len(comp_df) > 0:
        comp_df["os_event"] = comp_df["os_event"].astype(int)
        alive_paths = comp_df[(comp_df["os_event"] == 0) & (comp_df["cc_rank"] == 1)]["crop_path"].tolist()[:MONTAGE_N]
        dead_paths  = comp_df[(comp_df["os_event"] == 1) & (comp_df["cc_rank"] == 1)]["crop_path"].tolist()[:MONTAGE_N]

        alive_out = os.path.join(d_mont, "alive_top_attention_montage.png")
        dead_out  = os.path.join(d_mont, "dead_top_attention_montage.png")
        make_montage(alive_paths, alive_out, MONTAGE_COLS, MONTAGE_TILE)
        make_montage(dead_paths, dead_out, MONTAGE_COLS, MONTAGE_TILE)
        kv("Saved", alive_out)
        kv("Saved", dead_out)

    report_path = os.path.join(d_logs, "summary_report.txt")
    write_summary_txt(report_path, meta_w, enrich, global_metrics)
    kv("Saved", report_path)

    banner("FIN")
    kv("End", ts())
    kv("Run dir", run_dir)
    print("\nRevisa:")
    print(f"  - {run_dir}/02_cases/<CASE>/* (thumb/heat/mask/crops)")
    print(f"  - {run_dir}/03_quant/* (clusters + enrichment + group separation)")
    print(f"  - {run_dir}/04_montages/* (Alive/Dead montages)")
    print(f"  - {run_dir}/00_logs/summary_report.txt")

if __name__ == "__main__":
    main()
