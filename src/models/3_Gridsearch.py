import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import os
from pathlib import Path
import joblib

project_dir = Path(__file__).resolve().parents[2]

X_train = pd.read_csv(os.path.join(project_dir,"data/standardization/X_train.csv"))
Y_train = pd.read_csv(os.path.join(project_dir,"data/processed_data/y_train.csv"))

Y_train = np.ravel(Y_train)


model = GradientBoostingRegressor()
param_grid = {
    'n_estimators': [ 100, 200],
    'max_depth': [2,5],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [3,6]
}

grid_search = GridSearchCV(model, param_grid, cv=5)

grid_search.fit(X_train, Y_train)

joblib.dump(grid_search.best_params_, os.path.join(project_dir,"models/gradient_boosting_best_params.pkl"))