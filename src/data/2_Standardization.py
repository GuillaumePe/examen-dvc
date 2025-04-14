import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]

def save_dataframes(X_train, X_test, output_folderpath):
    check_existing_folder(output_folderpath)
    for file, filename in zip([X_train, X_test], ['X_train', 'X_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

    
def check_existing_folder(folder_path):
    '''Check if a folder already exists. If it doesn't, create it.'''
    if os.path.exists(folder_path) == False :
        os.makedirs(folder_path)

X_train = pd.read_csv(os.path.join(project_dir, "data/processed_data", 'X_train.csv'))
X_test = pd.read_csv(os.path.join(project_dir, "data/processed_data", 'X_test.csv'))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

save_dataframes(X_train_scaled, X_test_scaled, os.path.join(project_dir, "data/standardization"))