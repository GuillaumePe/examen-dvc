import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

db = pd.read_csv(os.path.join(project_dir,"/data/raw_data"))
X = db.drop(["silica_concentrate","date"])
Y = db["silica_concentrate"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
save_dataframes(X_train, X_test, y_train, y_test, os.path.join(project_dir,"/data/processed"))

