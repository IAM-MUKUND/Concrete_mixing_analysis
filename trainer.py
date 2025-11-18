import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from xgboost import XGBRegressor

print("torch.cuda.is_available():", torch.cuda.is_available())
use_gpu = torch.cuda.is_available()

df = pd.read_csv("/home/mukundvinayak/torch_installed_env/Concrete_mixing_analysis/preprocessed_concrete_mixing_entries.csv")
X = df.drop(columns=["compressive_strength(MPa)"])
y = df["compressive_strength(MPa)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_kwargs = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)


model = XGBRegressor(**xgb_kwargs)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

print("R² on test:", model.score(X_test, y_test))

out_dir = "./Concrete_mixing_analysis/model_outputs"
os.makedirs(out_dir, exist_ok=True)
model_path = os.path.join(out_dir, "xgb_concrete_model.json")
model.save_model(model_path)
import json
with open(model_path, 'r') as f:
    model_json = json.load(f)
with open(model_path, 'w') as f:
    json.dump(model_json, f, indent=2)

print("Saved model to", model_path)
