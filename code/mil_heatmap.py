# mkdir -p /home/facanor/Documents/PAAD
# cat > /home/facanor/Documents/PAAD/mil_heatmap_svs.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIL attention heatmap overlay & exports (v5)

Funciones clave:
- Reemplazo de prefijo en image_path (por defecto): "/home/jjlozanoj/NAS/Data" -> "/home/rdacostav/Documents/Data"
- Extracción robusta de (x,y) desde el nombre del parche (prioriza "_x<int>_y<int>")
- Manejo seguro de magnificación:
    * Lee magnificación base de propiedades del SVS si existe
    * Si no, asume 20X (más seguro que 40X)
    * Auto-selección del factor 20X->nivel-0 entre {base_mag/20, 1.0} según qué deja más parches dentro del slide
    * Opción --force-factor para forzar (p. ej. 1.0 si tus coords ya están en nivel-0)
- Asegura que el heatmap tenga EXACTAMENTE el mismo tamaño que el thumbnail
- Exporta por caso:
    * __thumb.png        (thumbnail original)
    * __thumb_dark.png   (thumbnail oscurecido)
    * __thumb_proc.png   (thumbnail oscurecido + blur + desaturado)
    * __heat.png         (overlay: thumbnail procesado + heat con modo de mezcla)
    * __heat_only.png    (solo heat, RGB)
    * __heat_rgba.png    (opcional: heat con alpha según intensidad)
    * __patchmap.png     (cobertura de TODOS los parches)
    * __heat.npy         (float32 [0,1])
    * __cover.npy        (float32 cuentas)
- Opcional --crop-to-cover para recortar todo al bbox de los parches

Controles de contraste:
- --heat-top-frac: el heat se construye con la fracción top de parches por caso (cobertura usa todos)
- --log-weights, --clip-low/--clip-high, --gamma: normalización robusta
- --cmap: usa mapas fríos (viridis/cividis/Blues) para contrastar con H&E
- --thumb-dim, --thumb-blur, --thumb-sat: oscurecer, desenfocar, (des)saturar el fondo
- --blend: alpha/screen/add/multiply/overlay (screen suele ir muy bien sobre H&E)
- --mask-quantile: pintar sólo el cuantil alto del heat
"""

import os
import re
import argparse
from glob import glob
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- OpenSlide --------------------
try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python es requerido. Instala `pip install openslide-python` y asegúrate de tener libopenslide.") from e

try:
    from skimage.filters import gaussian as sk_gaussian
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

# -------------------- Utilities --------------------
PATTERN_CANDIDATES = [
    re.compile(r"_x(?P<x>\d+)_y(?P<y>\d+)", re.IGNORECASE),
    re.compile(r"[^0-9]x(?P<x>\d+)[^0-9]y(?P<y>\d+)", re.IGNORECASE),
    re.compile(r"[^0-9\-]x(?P<x>-?\d+)[^0-9\-]y(?P<y>-?\d+)", re.IGNORECASE),
    re.compile(r"patch[_-](?P<x>\d+)[_-](?P<y>\d+)", re.IGNORECASE),
    re.compile(r"/(?P<x>\d+)[_-](?P<y>\d+)(?:\.\w+)?$", re.IGNORECASE),
]

def parse_xy_from_path(path: str):
    name = str(path)
    for pat in PATTERN_CANDIDATES:
        m = pat.search(name)
        if m:
            try:
                x = int(m.group("x")); y = int(m.group("y"))
                return x, y
            except Exception:
                continue
    raise ValueError(f"No pude extraer x,y del nombre: {path}")

def find_svs_for_case(case_id: str, svs_dir: str):
    patterns = [f"{case_id}*.svs", f"{case_id}*.tif", f"{case_id}*.tiff", f"{case_id}*.ndpi", f"{case_id}*.scn"]
    for pat in patterns:
        hits = glob(os.path.join(svs_dir, pat))
        if hits:
            hits.sort(key=lambda p: (len(Path(p).name), p))
            return hits[0]
    return None

def get_base_magnification(slide: "openslide.OpenSlide"):
    props = slide.properties
    for key in ["aperio.AppMag", "openslide.objective-power", "hamamatsu.SourceLens"]:
        if key in props:
            try:
                val = float(props[key])
                if 5 <= val <= 100:
                    return val
            except Exception:
                continue
    return 20.0  # fallback más seguro

def make_thumbnail(slide: "openslide.OpenSlide", max_side: int = 3000):
    w0, h0 = slide.dimensions
    if max(w0, h0) <= max_side:
        tw, th = w0, h0
    else:
        if w0 >= h0:
            tw = max_side; th = int(round(h0 * (max_side / w0)))
        else:
            th = max_side; tw = int(round(w0 * (max_side / h0)))
    scale = tw / w0
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    thumb = thumb.resize((tw, th))  # fuerza el tamaño exacto
    return thumb, scale

def determine_factor_20x_to_lvl0(slide_w0, slide_h0, coords_xy, patch_size_20x, base_mag, force_factor=None):
    if force_factor is not None:
        return float(force_factor)
    candidates = []
    if base_mag and base_mag > 0:
        candidates.append(float(base_mag) / 20.0)
    candidates.append(1.0)
    best, best_inside = candidates[0], -1
    sample = coords_xy[:min(200, len(coords_xy))] if isinstance(coords_xy, list) else list(coords_xy)[:min(200, len(coords_xy))]
    for fac in candidates:
        inside = 0
        ps = patch_size_20x * fac
        for (x20, y20) in sample:
            x0 = x20 * fac; y0 = y20 * fac
            if (0 <= x0 < slide_w0) and (0 <= y0 < slide_h0) and (x0 + ps <= slide_w0) and (y0 + ps <= slide_h0):
                inside += 1
        if inside > best_inside:
            best_inside, best = inside, fac
    return float(best)

def build_maps(coords_xy, weights, slide_w0, slide_h0, target_size,
               patch_size_20x=224, base_mag=40.0, sigma=1.5, force_factor=None):
    W, H = target_size
    if len(coords_xy) == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)
    factor_20x_to_lvl0 = determine_factor_20x_to_lvl0(slide_w0, slide_h0, coords_xy, patch_size_20x, base_mag, force_factor=force_factor)
    scale_x = W / float(slide_w0); scale_y = H / float(slide_h0)
    ps_lvl0 = float(patch_size_20x) * factor_20x_to_lvl0
    ps_thumb_x = max(1, int(round(ps_lvl0 * scale_x)))
    ps_thumb_y = max(1, int(round(ps_lvl0 * scale_y)))
    heat = np.zeros((H, W), dtype=np.float32)
    cover = np.zeros((H, W), dtype=np.float32)
    for (x20, y20), w in zip(coords_xy, weights):
        x0 = float(x20) * factor_20x_to_lvl0
        y0 = float(y20) * factor_20x_to_lvl0
        xt = int(round(x0 * scale_x)); yt = int(round(y0 * scale_y))
        x1 = max(0, xt); y1 = max(0, yt)
        x2 = min(W, xt + ps_thumb_x); y2 = min(H, yt + ps_thumb_y)
        if x1 < x2 and y1 < y2:
            heat[y1:y2, x1:x2] += float(w)
            cover[y1:y2, x1:x2] += 1.0
    if HAS_SKIMAGE and sigma > 0:
        heat = sk_gaussian(heat, sigma=sigma, preserve_range=True)
    if heat.max() > 0:
        heat = heat / (heat.max() + 1e-8)
    return heat.astype(np.float32), cover.astype(np.float32)

def darken_image(img: Image.Image, dim: float = 0.15) -> Image.Image:
    dim = float(max(0.0, min(dim, 1.0)))
    if dim == 0.0:
        return img
    arr = np.asarray(img).astype(np.float32)
    factor = 1.0 - dim
    arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def process_thumbnail(img: Image.Image, dim: float = 0.15, blur: float = 0.0, sat: float = 1.0) -> Image.Image:
    out = darken_image(img, dim=dim)
    if blur and blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(blur)))
    if sat != 1.0:
        enhancer = ImageEnhance.Color(out)
        out = enhancer.enhance(float(sat))
    return out

def blend_images(base_rgb: np.ndarray, heat_rgb: np.ndarray, alpha, mode: str = "alpha"):
    if isinstance(alpha, (float, int)):
        alpha_map = float(alpha)
    else:
        alpha_map = alpha  # HxWx1 o HxW
    if mode == "alpha":
        return (1 - alpha_map) * base_rgb + alpha_map * heat_rgb
    if mode == "screen":
        return 1 - (1 - base_rgb) * (1 - alpha_map * heat_rgb)
    if mode == "add":
        return np.clip(base_rgb + alpha_map * heat_rgb, 0, 1)
    if mode == "multiply":
        return np.clip(base_rgb * (1 - alpha_map + alpha_map * heat_rgb), 0, 1)
    if mode == "overlay":
        mask = base_rgb <= 0.5
        low = 2 * base_rgb * (1 - alpha_map + alpha_map * heat_rgb)
        high = 1 - 2 * (1 - base_rgb) * (1 - (1 - alpha_map + alpha_map * heat_rgb))
        return np.where(mask, low, high)
    return (1 - alpha_map) * base_rgb + alpha_map * heat_rgb

def colormap_overlay(rgb_img: Image.Image, heat: np.ndarray, alpha: float = 0.45, cmap_name: str = "viridis", mask_quantile: float = 0.0, blend: str = "screen"):
    H_img, W_img = rgb_img.size[1], rgb_img.size[0]
    if heat.shape[0] != H_img or heat.shape[1] != W_img:
        heat_img = Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8))
        heat_img = heat_img.resize((W_img, H_img), resample=Image.BILINEAR)
        heat = np.asarray(heat_img).astype(np.float32) / 255.0
    rgb = np.asarray(rgb_img).astype(np.float32) / 255.0
    if mask_quantile and mask_quantile > 0.0:
        thr = float(np.quantile(heat, min(max(mask_quantile, 0.0), 1.0)))
        mask = (heat >= thr).astype(np.float32)
        alpha_map = alpha * mask[..., None]
    else:
        alpha_map = alpha
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(heat)[..., :3]
    blended = blend_images(rgb, heat_rgb, alpha_map, mode=blend)
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

def render_heat_only(heat: np.ndarray, cmap_name: str = "viridis") -> Image.Image:
    cmap = plt.get_cmap(cmap_name)
    rgb = (cmap(heat)[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)

def render_heat_rgba(heat: np.ndarray, cmap_name: str = "viridis",
                     alpha_scale: float = 1.0, alpha_threshold: float = 0.0) -> Image.Image:
    heat = np.clip(heat, 0, 1).astype(np.float32)
    cmap = plt.get_cmap(cmap_name)
    rgb = cmap(heat)[..., :3]
    a = heat.copy()
    if alpha_threshold > 0:
        a[a < alpha_threshold] = 0.0
    a = np.clip(a * float(alpha_scale), 0, 1)
    rgba = np.dstack([rgb, a[..., None]])
    return Image.fromarray((rgba * 255.0).astype(np.uint8))

def render_patchmap(cover: np.ndarray, bg=(128, 128, 160)) -> Image.Image:
    H, W = cover.shape
    base = Image.new("RGB", (W, H), color=tuple(int(c) for c in bg))
    if cover.max() <= 0:
        return base
    mask = (cover > 0).astype(np.uint8) * 255
    white = Image.new("RGB", (W, H), color=(240, 240, 240))
    base_np = np.asarray(base).copy()
    mask3 = np.repeat(mask[..., None], 3, axis=2)
    out = np.where(mask3 == 255, np.asarray(white), base_np).astype(np.uint8)
    return Image.fromarray(out)

def normalize_weights(weights: np.ndarray, log_weights: bool = False,
                      clip_low: float = 0.0, clip_high: float = 100.0, gamma: float = 1.0):
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return w.astype(np.float32)
    if log_weights:
        w = np.log1p(w - np.min(w) + 1e-12)
    clip_low = float(max(0.0, min(clip_low, 100.0)))
    clip_high = float(max(0.0, min(clip_high, 100.0)))
    if clip_high < clip_low:
        clip_low, clip_high = clip_high, clip_low
    lo = np.percentile(w, clip_low) if clip_low > 0 else np.min(w)
    hi = np.percentile(w, clip_high) if clip_high < 100 else np.max(w)
    if hi > lo:
        w = np.clip(w, lo, hi)
    wmin, wmax = np.min(w), np.max(w)
    if wmax > wmin:
        w = (w - wmin) / (wmax - wmin)
    else:
        w = np.zeros_like(w, dtype=np.float64)
    if gamma != 1.0:
        w = np.power(w, gamma)
    return w.astype(np.float32)

# -------------------- Main --------------------
def main(args):
    df = pd.read_pickle(args.pkl)
    if not {"case_id", "image_path", "attention_weight"}.issubset(df.columns):
        raise ValueError("El PKL debe tener columnas: case_id, image_path, attention_weight")

    # Reemplazo de prefijo
    if args.path_replace_from or args.path_replace_to:
        if not (args.path_replace_from and args.path_replace_to):
            raise ValueError("Usa --path-replace-from y --path-replace-to juntos.")
        df["image_path"] = df["image_path"].astype(str).str.replace(args.path_replace_from, args.path_replace_to, n=1, regex=False)
    else:
        old_prefix = "/home/jjlozanoj/NAS/Data"
        new_prefix = "/home/rdacostav/Documents/Data"
        df["image_path"] = df["image_path"].astype(str).str.replace(old_prefix, new_prefix, n=1, regex=False)

    groups = {k: v for k, v in df.groupby("case_id")}

    map_df = None
    if args.map_csv:
        map_df = pd.read_csv(args.map_csv)
        if not {"case_id", "svs_path"}.issubset(map_df.columns):
            raise ValueError("El CSV de mapeo debe tener columnas: case_id, svs_path")

    os.makedirs(args.out, exist_ok=True)
    case_ids = sorted(set(df["case_id"].astype(str).tolist()))
    summary_rows = []

    for case in case_ids:
        if map_df is not None:
            subset = map_df[map_df["case_id"].astype(str) == case]
            svs_path = subset["svs_path"].iloc[0] if len(subset) else None
        else:
            svs_path = find_svs_for_case(case, args.svs_dir)

        if svs_path is None or not os.path.isfile(svs_path):
            warnings.warn(f"[{case}] SVS no encontrado. Salto el caso.")
            summary_rows.append({"case_id": case, "svs_found": False, "n_patches": len(groups.get(case, []))})
            continue

        patches = groups.get(case, None)
        if patches is None or len(patches) == 0:
            warnings.warn(f"[{case}] Sin parches en PKL. Exporto solo thumbnail.")
            patches = pd.DataFrame(columns=["image_path", "attention_weight"])

        slide = openslide.OpenSlide(svs_path)
        w0, h0 = slide.dimensions
        base_mag = get_base_magnification(slide)
        thumb, _ = make_thumbnail(slide, max_side=args.max_side)

        # Parse coords
        coords = []
        for pth in patches["image_path"].astype(str).tolist():
            try:
                coords.append(parse_xy_from_path(pth))
            except Exception as e:
                warnings.warn(str(e))
                coords.append(None)

        ok_mask = [c is not None for c in coords]
        coords_ok = [c for c in coords if c is not None]
        weights_all = patches["attention_weight"].to_numpy()[ok_mask]

        # Top-frac para HEAT
        if len(coords_ok) > 0:
            order = sorted(range(len(coords_ok)), key=lambda i: float(weights_all[i]), reverse=True)
            k = max(1, int(round(len(order) * min(max(args.heat_top_frac, 0.0), 1.0))))
            idx_top = order[:k]
            coords_top = [coords_ok[i] for i in idx_top]
            weights_top = weights_all[idx_top]
        else:
            coords_top, weights_top = [], []

        # Normalización robusta
        weights_top = normalize_weights(weights_top, log_weights=args.log_weights,
                                        clip_low=args.clip_low, clip_high=args.clip_high, gamma=args.gamma)

        # Mapas
        heat, _ = build_maps(coords_top, weights_top, w0, h0, thumb.size,
                             patch_size_20x=args.patch_size, base_mag=base_mag,
                             sigma=args.sigma, force_factor=args.force_factor)
        ones = np.ones(len(coords_ok), dtype=np.float32)
        _, cover = build_maps(coords_ok, ones, w0, h0, thumb.size,
                              patch_size_20x=args.patch_size, base_mag=base_mag,
                              sigma=0.0, force_factor=args.force_factor)

        # Thumbnail procesado: oscurecer -> blur -> (de)saturar
        thumb_dark = darken_image(thumb, dim=args.thumb_dim)
        thumb_proc = process_thumbnail(thumb, dim=args.thumb_dim, blur=args.thumb_blur, sat=args.thumb_sat)

        # Render
        out_overlay = thumb_proc if len(coords_ok) == 0 else colormap_overlay(
            thumb_proc, heat, alpha=args.alpha, cmap_name=args.cmap,
            mask_quantile=args.mask_quantile, blend=args.blend
        )
        heat_only_img = render_heat_only(heat, cmap_name=args.cmap)
        patchmap_img = render_patchmap(cover)

        # Crop opcional
        if args.crop_to_cover and cover.max() > 0:
            ys, xs = np.where(cover > 0)
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(cover.shape[0], y2), min(cover.shape[1], x2)
            heat = heat[y1:y2, x1:x2]
            cover = cover[y1:y2, x1:x2]
            thumb = thumb.crop((x1, y1, x2, y2))
            thumb_dark = thumb_dark.crop((x1, y1, x2, y2))
            thumb_proc = thumb_proc.crop((x1, y1, x2, y2))
            out_overlay = out_overlay.crop((x1, y1, x2, y2))
            heat_only_img = heat_only_img.crop((x1, y1, x2, y2))
            patchmap_img = patchmap_img.crop((x1, y1, x2, y2))

        # Guardado
        base = Path(svs_path).stem
        out_overlay_png     = os.path.join(args.out, f"{case}__{base}__heat.png")
        out_thumb_png       = os.path.join(args.out, f"{case}__{base}__thumb.png")
        out_thumb_dark_png  = os.path.join(args.out, f"{case}__{base}__thumb_dark.png")
        out_thumb_proc_png  = os.path.join(args.out, f"{case}__{base}__thumb_proc.png")
        out_heat_only_png   = os.path.join(args.out, f"{case}__{base}__heat_only.png")
        out_patchmap_png    = os.path.join(args.out, f"{case}__{base}__patchmap.png")
        out_heat_npy        = os.path.join(args.out, f"{case}__{base}__heat.npy")
        out_cover_npy       = os.path.join(args.out, f"{case}__{base}__cover.npy")

        out_overlay.save(out_overlay_png, quality=95)
        thumb.save(out_thumb_png, quality=95)
        thumb_dark.save(out_thumb_dark_png, quality=95)
        thumb_proc.save(out_thumb_proc_png, quality=95)
        heat_only_img.save(out_heat_only_png, quality=95)
        patchmap_img.save(out_patchmap_png, quality=95)
        np.save(out_heat_npy, heat.astype(np.float32))
        np.save(out_cover_npy, cover.astype(np.float32))

        heat_rgba_png = None
        if args.export_heat_rgba:
            heat_rgba_png = os.path.join(args.out, f"{case}__{base}__heat_rgba.png")
            heat_rgba_img = render_heat_rgba(heat, cmap_name=args.cmap,
                                             alpha_scale=args.heat_alpha_scale,
                                             alpha_threshold=args.heat_alpha_threshold)
            heat_rgba_img.save(heat_rgba_png)

        summary_rows.append({
            "case_id": case,
            "svs_found": True,
            "n_patches": int(len(coords_ok)),
            "svs_path": svs_path,
            "thumb_png": out_thumb_png,
            "thumb_dark_png": out_thumb_dark_png,
            "thumb_proc_png": out_thumb_proc_png,
            "overlay_png": out_overlay_png,
            "heat_only_png": out_heat_only_png,
            "heat_rgba_png": heat_rgba_png,
            "patchmap_png": out_patchmap_png,
            "heat_npy": out_heat_npy,
            "cover_npy": out_cover_npy,
            "base_mag": base_mag,
            "thumb_size": f"{thumb.size[0]}x{thumb.size[1]}"
        })

        slide.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Done. Summary: {summary_csv}")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay MIL attention heatmaps on SVS thumbnails and export maps (v5).")
    parser.add_argument("--pkl", required=True, help="Path to PKL con columnas [case_id, image_path, attention_weight, (label opcional)].")
    parser.add_argument("--svs-dir", required=True, help="Directorio con SVS/WSI.")
    parser.add_argument("--out", required=True, help="Directorio de salida.")
    parser.add_argument("--map-csv", default=None, help="CSV opcional {case_id, svs_path}.")
    parser.add_argument("--path-replace-from", default=None, help="Prefijo antiguo en image_path.")
    parser.add_argument("--path-replace-to",   default=None, help="Prefijo nuevo en image_path.")
    parser.add_argument("--max-side", type=int, default=3000, help="Lado máximo del thumbnail.")
    parser.add_argument("--patch-size", type=int, default=224, help="Tamaño de parche a 20X (px).")
    parser.add_argument("--sigma", type=float, default=1.5, help="Sigma del suavizado del heatmap.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Opacidad del overlay [0..1].")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap (viridis, cividis, Blues, magma, turbo, jet...).")
    parser.add_argument("--log-weights", action="store_true", help="log1p antes de normalizar pesos.")
    parser.add_argument("--clip-low", type=float, default=0.0, help="Percentil bajo para recorte de pesos (0-100).")
    parser.add_argument("--clip-high", type=float, default=100.0, help="Percentil alto para recorte de pesos (0-100).")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma en pesos normalizados (<1 potencia bajos).")
    parser.add_argument("--heat-top-frac", type=float, default=1.0, help="Fracción top (0-1] de parches para HEAT por caso (cobertura usa todos).")
    parser.add_argument("--force-factor", type=float, default=None, help="Forzar factor 20X->nivel-0 (1.0 si ya están en nivel-0).")
    parser.add_argument("--crop-to-cover", dest="crop_to_cover", action="store_true", help="Recorta al bbox de cobertura.")
    parser.add_argument("--mask-quantile", type=float, default=0.0, help="Pintar sólo pixels >= a este cuantil [0-1].")
    parser.add_argument("--blend", type=str, default="screen", choices=["alpha","screen","add","multiply","overlay"], help="Modo de mezcla.")
    parser.add_argument("--export-heat-rgba", action="store_true", help="Exportar también __heat_rgba.png con alpha del heat.")
    parser.add_argument("--heat-alpha-scale", type=float, default=1.0, help="Escala de alpha para __heat_rgba.png.")
    parser.add_argument("--heat-alpha-threshold", type=float, default=0.0, help="Umbral de alpha para __heat_rgba.png.")
    parser.add_argument("--thumb-dim", type=float, default=0.15, help="Oscurecer thumbnail esta fracción (0-1).")
    parser.add_argument("--thumb-blur", type=float, default=1.5, help="Blur gaussiano (px) al thumbnail.")
    parser.add_argument("--thumb-sat", type=float, default=0.9, help="Saturación del thumbnail (1=igual, 0=gris).")
    args = parser.parse_args()
    raise SystemExit(main(args))
PY
chmod +x /home/facanor/Documents/PAAD/mil_heatmap_svs.py
