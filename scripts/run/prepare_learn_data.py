#! python3
# -*- coding: UTF-8 -*-

# %% Импорт
import os
import warnings
import argparse
import mlflow
from mlflow.tracking import MlflowClient

from config import settings

import scripts.data_scripts.prepare_data as prep
import scripts.utils.files as files

warnings.simplefilter("ignore")

# %%
stage_name = 'prepare_learn_data'
dtypes = settings.get_fresh('types.CSV_DTYPES')
subdir = settings.get_fresh('PROCESSED_LOCAL_SUBDIR')
cleaned_data_file = 'cleaned'
use_mlflow = False

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Имя файла с датасетом для обработки')
    parser.add_argument('-m', '--mlflow', action='store_true', help='Использовать MlFlow')
    namespace = parser.parse_args()
    if namespace.file:
        cleaned_data_file = namespace.file
    if namespace.mlflow == True:
        use_mlflow = True

# %%
if use_mlflow:
    # Настройки MLFlow
    mlflow_dir = settings.get_fresh('MLFLOW_DIR')
    mlflow_set_tracking_uri = settings.get_fresh('MLFLOW_SET_TRACKING_URI')
    os.environ["MLFLOW_REGISTRY_URI"] = mlflow_dir
    mlflow.set_tracking_uri(mlflow_set_tracking_uri)
    mlflow.set_experiment(stage_name)

# %%
print('Начинается загрузка файла данных')
df = files.load_file(cleaned_data_file, subdir, dtypes=dtypes, to_decompress=True)
print('Начинается обработка файла данных')
new_df = prep.prepare_data_for_parse(df)

# %%
print('Начинается сохранение файла данных')
filename = settings.get_fresh('FILENAME_PREPROCESSED')
processed_data_file_path = files.save_dataframe(new_df, filename, subdir)
print(f'Файл {processed_data_file_path} сохранен')

# %%
if use_mlflow:
    # Логирование в MLFlow
    script_file_path = files.get_run_script_fullpath(stage_name)
    with mlflow.start_run():
        mlflow.log_artifact(local_path=script_file_path, artifact_path=stage_name)
        mlflow.log_artifact(local_path=processed_data_file_path, artifact_path=stage_name)
        mlflow.end_run()

print('Выполнение закончено')