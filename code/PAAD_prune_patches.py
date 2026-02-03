"""
This script was used for the TCGA 40-case experiment.
Paths may need to be adapted for different environments.
"""
import re
import os
import shutil
import numpy as np
from scipy.ndimage import label
from pathlib import Path

# Paths

patches_path = "/	


def prune_saved_patches(folder, patch_size, connectivity=8, min_patches=50):
    coords = []
    file_map = {}

    for fname in os.listdir(folder):
        m = re.search(r'_x(\d+)_y(\d+)\.jpg$', fname)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            coords.append((x, y))
            file_map[(x, y)] = fname

    if not coords:
        print(f"No matching patch files in {folder}")
        return

    xs = [x // patch_size for x, _ in coords]
    ys = [y // patch_size for _, y in coords]
    max_x, max_y = max(xs), max(ys)

    grid = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
    for gx, gy in zip(xs, ys):
        grid[gy, gx] = 1

    if connectivity == 8:
        structure = np.ones((3,3), dtype=np.int32)
    else:
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.int32)

    labeled, _ = label(grid, structure=structure)
    sizes = np.bincount(labeled.ravel())
    keep_labels = set(np.where(sizes >= min_patches)[0])
    keep_labels.discard(0)
   
    removed_dir = Path(folder) / "Removed"
    removed_dir.mkdir(exist_ok=True)
 
    kept, removed = 0, 0
    for (x, y), fname in file_map.items():
        gx, gy = x // patch_size, y // patch_size
        src = Path(folder) / fname
        dst = removed_dir / fname
        if labeled[gy, gx] not in keep_labels:
            shutil.move(str(src), str(dst))
            removed += 1
        else:
            kept += 1

    print(f"{folder}: {kept} kept, {removed} removed")

def prune_all_subfolders(patches_path, patch_size, connectivity=8, min_patches=50):
    for root, dirs, _ in os.walk(patches_path):
        for d in dirs:
            slide_folder = os.path.join(root, d)
            if d == "Removed":  # skip removed subfolders
                continue
            print(f"{slide_folder}")
            prune_saved_patches(slide_folder, patch_size, connectivity, min_patches)



prune_all_subfolders(patches_path, patch_size=224, connectivity=8, min_patches=40)

# nohup python -u PAAD_prune_patches.py > PAAD_prune_patches.txt

