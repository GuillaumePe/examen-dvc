import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]

def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

scaler = StandardScaler()

scaled_data_dict = {}
for filename in ['X_train', 'X_test']:
    data = pd.read_csv(os.path.join(project_dir,"/data/processed", f'{filename}.csv'))
    data_scaled = scaler.ft_transform(data)
    scaled_data_dict[filename] = data_scaled
save_dataframes(scaled_data_dict['X_train'], scaled_data_dict['X_test'], os.path.join(project_dir,"/data/processed"))