import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    check_existing_folder(output_folderpath)
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

    
def check_existing_folder(folder_path):
    '''Check if a folder already exists. If it doesn't, create it.'''
    if os.path.exists(folder_path) == False :
        os.makedirs(folder_path)

db = pd.read_csv(os.path.join(project_dir,"data/raw_data/raw_data.csv"))
#print(db.columns)
X = db.drop(["silica_concentrate","date"], axis=1)
Y = db["silica_concentrate"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
save_dataframes(X_train, X_test, y_train, y_test, os.path.join(project_dir,"data/processed_data"))



