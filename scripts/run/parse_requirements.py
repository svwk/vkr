#! python3
# -*- coding: UTF-8 -*-

# %% Импорт
import os
import warnings
import argparse
import mlflow
from mlflow.tracking import MlflowClient

from config import settings

import scripts.model_scripts.text_process as proc
import scripts.utils.files as files
import scripts.utils.tasks as tasks

warnings.simplefilter("ignore")

# %%
stage_name = 'parse_requirements'
model_subdir = settings.get_fresh('BLOCK_MODELING_SUBDIR')
processed_subdir = settings.get_fresh('PROCESSED_LOCAL_SUBDIR')
data_file = settings.get_fresh('FILENAME_BLOCKS_SELECTED')
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
df = files.load_file(data_file, processed_subdir,  to_decompress=True, with_dates=False)
print('Начинается обработка файла данных')
new_df = tasks.apply_parallel(df, proc.parse_requirements)
# new_df = proc.parse_requirements(df)

# %%
print('Начинается сохранение файла данных')
filename = settings.get_fresh('FILENAME_REQUIREMENT_EXTRACTED')
processed_data_file_path = files.save_dataframe(new_df, filename, model_subdir)
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



