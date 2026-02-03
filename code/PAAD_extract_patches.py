import os
import fnmatch
import numpy as np
from PIL import Image
import large_image
import cv2
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Process WSI images and extract patches.')
    parser.add_argument('-i', '--images_path', type=str, required=True, help='Path to the images directory.')
    parser.add_argument('-p', '--patches_path', type=str, required=True, help='Path to the patches directory.')
    parser.add_argument('-s', '--patch_size', type=int, default=256, help='Size of the patches to be extracted.')
    parser.add_argument('-m', '--magnification', type=int, default=40, help='Magnification of the patches to be extracted.')
    return parser.parse_args()

def load_dataset(images_path):
    return [os.path.join(root, f)
            for root, _, files in os.walk(images_path)
            for f in files if f.lower().endswith(".svs")]

def tissue_mask_hsv(tile, s_thresh=0.07, v_thresh=0.94):
    #s_thresh=0.07, v_thresh=0.94 was my original threshold 
    """
    Returns True if tile has enough tissue based on HSV thresholds.
    tile: NumPy array (H, W, 3) in RGB format.
    s_thresh: Minimum saturation (0-1) to consider tissue.
    v_thresh: Maximum value/brightness (0-1) to consider tissue.
    """
    # Convert to HSV and normalize
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV) / 255.0
    s_mean = hsv[:, :, 1].mean()
    v_mean = hsv[:, :, 2].mean()

    return (s_mean > s_thresh) and (v_mean < v_thresh)

def extract_patches_hsv(image_path, patches_path, patch_size, magnification,s_thresh=0.04, v_thresh=0.96):
    #s_thresh=0.04, v_thresh=0.96 is my current treshold, it detects better the adipose and stroma regions, altough it can be too sensitive and detect some background
    image = large_image.open(image_path)
    image_name = Path(image_path).stem

    level = image.getLevelForMagnification(magnification)
    metadata = image.getMagnificationForLevel(level=level)
    sizeX, sizeY = image.getMetadata()['sizeX'], image.getMetadata()['sizeY']

    out_dir = Path(patches_path) / image_name
    out_dir.mkdir(parents=True, exist_ok=True)

    count, skipped, rejected = 0, 0, 0

    for y in range(0, sizeY, patch_size):
        for x in range(0, sizeX, patch_size):

            fname = f"{image_name}_{magnification}x_{patch_size}px_x{x}_y{y}.jpg" 
            fpath = out_dir / fname 
            removed_path = out_dir / "Removed" / fname

            # Skip if patch already exists or was marked removed 
            if fpath.exists() or removed_path.exists(): 
                skipped += 1 
                continue
   
            tile_info = image.getRegion(
                region=dict(left=x, top=y, width=patch_size, height=patch_size, units='base_pixels'),
                scale=dict(magnification=magnification),
                format=large_image.constants.TILE_FORMAT_NUMPY
            )
            tile = tile_info[0]
            if tissue_mask_hsv(tile,s_thresh,v_thresh):
                
                patch = Image.fromarray(tile).convert("RGB")
                patch.save(fpath)
                count += 1
            else: rejected += 1
    print(f"{image_name}: {count} new patches saved, "
          f"{skipped} skipped, {rejected} rejected.")


def main(images_path, patches_path, patch_size, magnification, sites):
    images = load_dataset(images_path)
    print(f"Loaded dataset\nImages: {len(images)}")

    os.makedirs(patches_path, exist_ok=True)

    for idx, image_path in enumerate(images, start=1):
        print(f"Tracking: {idx} of {len(images)} images\nRoute: {image_path}")
        for site in sites:
            try:
                extract_patches_hsv(image_path, patches_path,patch_size,magnification)
            except Exception as e:
                print(f"Error processing site {site} for image {image_name}: {e}")
        print(f"End of {image_path}\n")

if __name__ == "__main__":
    args = parse_args()
    sites = [1]
    main(args.images_path, args.patches_path, args.patch_size, args.magnification, sites)

# nohup python -u PAAD_extract_patches.py --images_path /home/jjlozanoj/NAS/Data/PAAD/Images --patches_path /home/jjlozanoj/NAS/Data/PAAD/Patches --patch_size 224 --magnification 20 > PAAD_extract_patches.txt

#images_path = "/content/drive/MyDrive/PDAC/IMAGES/Infiltrating duct carcinoma, NOS/"   # folder with .svs
#patches_path = "/content/drive/MyDrive/PDAC/PATCHES"
#patch_size = 224
#magnification = 20