import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import os
from pathlib import Path
import joblib

project_dir = Path(__file__).resolve().parents[2]

X_train = pd.read_csv(os.path.join(project_dir,"/data/processed/X_train.csv"))
Y_train = pd.read_csv(os.path.join(project_dir,"/data/processed/y_train.csv"))

best_params = joblib.load(os.path.join(project_dir,"models/gradient_boosting_best_params.pkl"))

model = GradientBoostingRegressor(**best_params)
model.fit(X_train, Y_train)

joblib.dump(model,os.path.join(project_dir,"models/gradient_boosting_best_model.pkl"))
