import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import glob
import sys
#from sklearn.metrics import accuracy_score, roc_auc_score


####
Test case, just to load one of the latent part files
####

#df_test = pd.read_pickle("/home/jjlozanoj/NAS/Data/PAAD/Latents/latents_part1.pkl")
# Pick a single case_id
#case_ids = df_test["image_path"].apply(lambda x: os.path.basename(x).split("_DX")[0]).unique()[:2]
#print("Testing case_id:", case_ids)
# Subset just those cases
#df_latents = df_test[df_test["image_path"].str.contains("|".join(case_ids))]
# Select first 10 embeddings (patches)
#df_latents = df_latents.groupby(df_latents["image_path"].apply(lambda x: os.path.basename(x).split("_DX")[0])).head(10).reset_index(drop=True)
#print("✅ Using", len(df_latents), "patches from case", case_id)
#df_latents["case_id"] = df_latents["image_path"].apply(
#    lambda x: os.path.basename(x).split("_DX")[0]
#)
#print("✅ Using", df_latents["case_id"].nunique(), "cases with", len(df_latents), "patches total")
#print(df_latents.head())


# Find all parts in the folder
#latent_files = sorted(glob.glob("/home/jjlozanoj/NAS/Data/PAAD/Latents/latents_part*.pkl"))

#print("Found latent files:", latent_files)

# Load and concat
#dfs = [pd.read_pickle(f) for f in latent_files]
#df_latents = pd.concat(dfs, ignore_index=True)

#print("✅ Combined DataFrame shape:", df_latents.shape)
#print(df_latents.head())


#df_latents["case_id"] = df_latents["image_path"].apply(
#    lambda x: os.path.basename(x).split("_DX")[0]
#)

#clinical = pd.read_csv(
#    " /home/jjlozanoj/NAS/Data/PAAD/filtered_clinical_data.csv",
#    usecols=["Patient ID", "Overall Survival Status", "Overall Survival (Months)"]
#)
#clinical["label"] = clinical["Overall Survival Status"].apply(lambda x: 1 if x == "0:LIVING" else 0)
#df_merged = df_latents.merge(clinical, left_on="case_id", right_on="Patient ID")



df_merged = pd.read_pickle("/home/jjlozanoj/NAS/Data/PAAD/Latents/valid-latents.pkl")
print("✅ Using valid latents only:", df_merged.shape)
print(df_merged.head())

class MILDataset(Dataset):
    def __init__(self, df, label_col="label"):
        self.patients = df["case_id"].unique()
        self.df = df
        self.label_col = label_col

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        case_id = self.patients[idx]
        bag = self.df[self.df["case_id"] == case_id]

        feature_cols = [c for c in bag.columns if isinstance(c, int)]  # only numeric feature cols
        X = bag[feature_cols].values
        X = torch.tensor(X, dtype=torch.float32)

        y = bag[self.label_col].iloc[0]
        y = torch.tensor(y, dtype=torch.long)
        paths = bag["image_path"].tolist()      

        return X, y,case_id,paths

patient_ids = df_merged["case_id"].unique()

np.random.seed(42)
np.random.shuffle(patient_ids)
split = int(0.8 * len(patient_ids))
train_ids, val_ids = patient_ids[:split], patient_ids[split:]


train_df = df_merged[df_merged["case_id"].isin(train_ids)]
val_df   = df_merged[df_merged["case_id"].isin(val_ids)]


train_dataset = MILDataset(train_df)
val_dataset   = MILDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1,shuffle=False)

class AttentionMIL(nn.Module):
    def __init__(self, input_dim=1536,hidden_attention_dim=512, output_dim=2,dropout=0.5, T=1.0):
# why only 2 layers here -> try adding more layers for smoothing
# Add dropout to the layers, so that is not overfitted
        super(AttentionMIL, self).__init__()
        # Attention Module
        self.attention_V = nn.Linear(input_dim,hidden_attention_dim)
        #self.attention_U = nn.Linear(input_dim,hidden_attention_dim)
        self.attention_weights = nn.Linear(hidden_attention_dim,1)
        
        self.classifier = nn.Sequential(
            #nn.Linear(input_dim, 1024),
            #nn.ReLU(),
            #nn.Dropout(dropout),
            
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            #nn.Linear(512, 256),
            #nn.ReLU(),
            #nn.Dropout(dropout),
            
            #nn.Linear(256, 128),
            #nn.ReLU(),
            #nn.Dropout(dropout),
            
            nn.Linear(512, output_dim)
        )
        self.T = T  # 🔥 temperature

    def forward(self, x):
        # Attention pooling
        A_V = torch.tanh(self.attention_V(x))
        #A_U = torch.sigmoid(self.attention_U(x))
        A = self.attention_weights(A_V) #* A_U)  # [N, 1]
        # Apply temperature scaling before softmax, just T=1.0 for now
        A = torch.softmax(A/self.T, dim=0)            # attention over patches
        M = torch.sum(A * x, dim=0)            # weighted average
   
        # Classification
        logits = self.classifier(M)
        return logits, A


def balanced_accuracy(y_true, y_pred, num_classes=2):
    recalls = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
    return np.mean(recalls) if recalls else 0.0

def roc_auc_score_np(y_true, y_score):
    # y_score = predicted probability for class 1
    desc = np.argsort(-y_score)
    y_true = np.array(y_true)[desc]
    y_score = np.array(y_score)[desc]

    # True positives and false positives
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    # Normalize
    tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
    fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps)

    # AUC via trapezoid rule
    auc = np.trapezoid(tpr, fpr)
    return auc

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
model = AttentionMIL(input_dim=1536).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
  
for epoch in tqdm(range(1, num_epochs + 1), desc="Training epochs"):
    model.train()
    train_losses = []
    print(f"\n🔄 Epoch {epoch}/{num_epochs}")

    for batch_idx, (X, y, _, _) in  enumerate(train_loader):
        X, y = X.squeeze(0).to(device), y.squeeze(0).to(device)
        out, A = model(X)
        loss = criterion(out.unsqueeze(0), y.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    tqdm.write(f"Epoch {epoch}/{num_epochs} - Train Loss: {np.mean(train_losses):.4f}")

    # ---- Collect attention weights for training set ----
    

    model.eval() 
    train_attn_results = []
    with torch.no_grad():
        for batch_idx, (X, y, case_id, paths) in enumerate(train_loader):
            X, y = X.squeeze(0).to(device), y.squeeze(0).to(device)
            out, A = model(X)

            case_id = case_id[0]
            label = y.item()
        
            A = A.squeeze().cpu().numpy()
            for p, w in zip(paths, A):
                train_attn_results.append({
                    "case_id": case_id,
                    "image_path": p,
                    "attention_weight": float(w),
                    "label": label
                })

    df_train_attn = pd.DataFrame(train_attn_results)
    df_train_attn.to_pickle(f"attention_weights_train_epoch{epoch}.pkl")
    tqdm.write(f"✅ Saved train patch-level attention weights for epoch {epoch}")


    val_attn_results,y_true, y_pred, y_prob = [], [], [], []

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y, case_id, paths) in enumerate(val_loader):

            X, y = X.squeeze(0).to(device), y.squeeze(0).to(device)
            out, A = model(X)

            # ---- Prediction ----
            prob = torch.softmax(out, dim=-1).cpu().numpy().flatten()
            pred = prob.argmax()
 
            y_true.append(y.item())
            y_pred.append(pred)
            y_prob.append(prob[1])
            case_id = case_id[0]
            label = y.item()
        
            # ---- Attention ----
            A = A.squeeze().cpu().numpy()

            for p, w in zip(paths, A):
                val_attn_results.append({
                    "case_id": case_id,
                    "image_path": p,
                    "attention_weight": float(w),
                    "label": label
                })

    # metrics
    val_acc = (np.array(y_true) == np.array(y_pred)).mean()
    val_bal_acc = balanced_accuracy(np.array(y_true), np.array(y_pred))
    val_auc = roc_auc_score_np(np.array(y_true), np.array(y_prob))

    tqdm.write(f"Val Acc: {val_acc:.4f}, Bal Acc: {val_bal_acc:.4f}, AUC: {val_auc:.4f}")

    df_attn = pd.DataFrame(val_attn_results)
    df_attn.to_pickle(f"attention_weights_val_epoch{epoch}.pkl")
    tqdm.write(f"✅ Saved patch-level attention weights for epoch {epoch}")

#nohup python mila_v2.py> mila_v2.txt
