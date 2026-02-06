import pandas as pd
import numpy as np

df_train = pd.read_pickle("attention_weights_train.pkl")
print(df_train.head())

print(np.max(df_train['attention_weight']))
print(np.min(df_train['attention_weight']))

df_val = pd.read_pickle("attention_weights_val.pkl")
print(df_val.head())

print(df_val['image_path'][0])