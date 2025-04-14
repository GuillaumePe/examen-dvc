from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
import numpy as np
from pathlib import Path
import joblib
import json

project_dir = Path(__file__).resolve().parents[2]

X_test = pd.read_csv(os.path.join(project_dir,"/data/processed/X_test.csv"))
Y_test = pd.read_csv(os.path.join(project_dir,"/data/processed/y_test.csv"))
Y_test = np.ravel(Y_test)

model = joblib.load(os.path.join(project_dir,"models/gradient_boosting_best_model.pkl"))
predictions = model.predict(X_test)

predictions.to_csv(os.path.join(project_dir,"/data/Predictions.csv"), index=False)

mean_squared_error_score = mean_squared_error(Y_test,predictions)
r2 = r2_score(Y_test, predictions)

metrics = {"mean_squared_error": mean_squared_error_score,"r2": r2}
metrics_path = os.path.join(project_dir,"metrics/scores.json")
metrics_path.write_text(json.dumps(metrics))

