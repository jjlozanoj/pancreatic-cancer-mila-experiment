import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse

# ----------------------------
# Args (choose which split to run)
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, required=True, help="Which split part to process (1-4)")
args = parser.parse_args()
part = args.part

# 🔑 Paste your Hugging Face token here
login(token="hf_xxxxxx")

timm_kwargs = {
    'img_size': 224,
    'patch_size': 14,
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5,
    'embed_dim': 1536,
    'mlp_ratio': 2.66667*2,
    'num_classes': 0,
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked,
    'act_layer': torch.nn.SiLU,
    'reg_tokens': 8,
    'dynamic_img_size': True
}

dmodel = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Using device:", device)


# ----------------------------
# Dataset
# ----------------------------	

class PatchDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

# ----------------------------
# Load this split
# ----------------------------

df = pd.read_pickle(f"patches_cases_ordered_part{part}.pkl")
print(f"✅ Loaded {len(df)} patches from part {part}")


dataset = PatchDataset(df, transform=transform)
loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)


# ----------------------------
# Extract latents
# ----------------------------

	

all_z = []
all_paths = []

with torch.no_grad():
    for images, paths in tqdm(loader, desc="Extracting latents", unit="batch"):
        images = images.to(device)sw
        z = model(images)                # forward pass
        z = z.cpu().numpy()              # → numpy
        all_z.extend(z)
        all_paths.extend(paths)

# ----------------------------
# Save
# ----------------------------

df_latents = pd.DataFrame(all_z)
df_latents["image_path"] = all_paths

os.makedirs("latents", exist_ok=True)
df_latents.to_pickle(f"latents/latents_part{part}.pkl")

print(f"✅ Saved latent vectors to latents/latents_part{part}.pkl")

print(df_latents.head())

# nohup python foundational_latents_uni2-h.py --part 1 > foundational_latents_uni2-h.txt 2>&1 &
# (later)
# nohup python foundational_latents_uni2-h.py --part 2 > foundational_latents_uni2-h.txt 2>&1 &
#(later)
# nohup python foundational_latents_uni2-h.py --part 3 > foundational_latents_uni2-h.txt 2>&1 &
#(later)
# nohup python foundational_latents_uni2-h.py --part 4 > foundational_latents_uni2-h.txt 2>&1 &

