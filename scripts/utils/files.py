import os
import pickle
import pandas as pd

from config import settings


def load_files(file_name_list, target_subdir, dtypes=None, to_decompress=True, with_dates=True):
    """
    Загрузка файлов с данными
    :param file_name_list: Список названий файлов
    :param target_subdir: Подкаталог для определенного типа датасетов в хранилище для данных
    :param dtypes: Словарь типов данных для загружаемого датасета
    :param to_decompress: Распаковать файл из архива
    :return: Список загруженных датафреймов
    """
    dfs = {}
    for file_name in file_name_list:
        df_part = load_file(file_name, target_subdir, dtypes=dtypes, to_decompress=to_decompress, with_dates=with_dates)
        if df_part is not None:
            dfs[file_name] = df_part

    return dfs


def load_file(file_name, target_subdir, dtypes=None, to_decompress=True, with_dates=True):
    """
    Загрузка файла с данными
    :param file_name: Название файла
    :param target_subdir: Подкаталог для определенного типа датасетов в хранилище для данных
    :param dtypes: Словарь типов данных для загружаемого датасета
    :param to_decompress: Распаковать файл из архива
    :return: Ссылка на загруженный датафрейм или None
    """

    file_fullpath = get_file_fullpath(file_name, 'd', target_subdir, is_compressed=to_decompress)
    parse_dates = ['published_at'] if with_dates is True else False

    try:
        dataframe = pd.read_csv(file_fullpath, low_memory=False, dtype=dtypes, parse_dates=parse_dates)
        return dataframe
    except:
        return file_fullpath

def load_data_dump(file_name, target_subdir):
    """
    Загрузка файла с дампом данных из хранилища для данных
    :param file_name: Название файла
    :param target_subdir: Подкаталог для определенного типа датасетов в хранилище для данных
    :return: Ссылка на загруженную модель или None
    """
    file_fullpath = get_file_fullpath(file_name, 'd', target_subdir, is_compressed=False, ext='dmp')
    try:
        with open(file_fullpath, "rb") as fd:
            model = pickle.load(fd)

        return model

    except:
        return file_fullpath


def load_model(file_name):
    """
    Загрузка файла с моделью из хранилища моделей
    :param file_name: Название файла
    :param target_subdir: Подкаталог для определенного типа датасетов в хранилище для данных
    :return: Ссылка на загруженную модель или None
    """
    file_fullpath = get_file_fullpath(file_name, 'm', None, is_compressed=False, ext='dmp')
    try:
        with open(file_fullpath, "rb") as fd:
            model = pickle.load(fd)

        return model

    except Exception as e:
        print(e)
        return file_fullpath


def save_dataframe(dataframe, file_name, target_subdir, to_compress=True, index=False):
    """
    Сохранение датафрейма в файл
    :param dataframe: Датафрейм для сохранения
    :param file_name: Название файла
    :param target_subdir: Подкаталог для определенного типа файлов в хранилище для данных
    :param is_compressed: Флаг упаковки результирующего файла в архив
    :return: Путь до сохраненного файла
    """

    file_fullpath = get_file_fullpath(file_name, 'd', target_subdir, is_compressed=to_compress)
    number = 1

    while os.path.exists(file_fullpath):
        number += 1
        file_fullpath = get_file_fullpath(file_name, 'd', target_subdir, number, is_compressed=to_compress)

    dataframe.to_csv(file_fullpath, index=index)

    return file_fullpath


def save_data_dump(model, file_name, target_subdir):
    """
    Сохранение дампа данных в файл в хранилище для данных
    :param file_name: Название файла
    :param target_subdir: Подкаталог в хранилище для данных
    :return: Путь до сохраненного файла
    """

    file_fullpath = get_file_fullpath(file_name, 'd', target_subdir, is_compressed=False, ext='dmp')
    number = 1

    while os.path.exists(file_fullpath):
        number += 1
        file_fullpath = get_file_fullpath(file_name, 'd', target_subdir, number, is_compressed=False, ext='dmp')

    with open(file_fullpath, "wb") as fd:
        pickle.dump(model, fd)

    return file_fullpath


def save_model(model, file_name):
    """
    Сохранение модели в файл в хранилище для моделей
    :param file_name: Название файла
    :return: Путь до сохраненного файла
    """

    file_fullpath = get_file_fullpath(file_name, 'm', None,  is_compressed=False, ext='dmp')
    number = 1

    while os.path.exists(file_fullpath):
        number += 1
        file_fullpath = get_file_fullpath(file_name, 'm', None, number,  is_compressed=False, ext='dmp')

    with open(file_fullpath, "wb") as fd:
        pickle.dump(model, fd)

    return file_fullpath


def get_file_fullpath(file_name, target_dir_type, target_subdir, number=None, is_compressed=True, ext = None):
    """
    Конструктор полного имени файла с данными
    :param file_name: Название файла с данными
    :param target_dir_type: Тип хранилища для определенного типа данных ('d','m')
    :param target_subdir: Подкаталог в хранилище для определенного типа данных
    :param number: Номер, добавляемый к названию файла
    :param is_compressed: Флаг упаковки результирующего файла в архив
    :return: Полный путь до файла
    """

    if target_dir_type == 'd':
        root_path = all_data_path
    elif target_dir_type == 'm':
        root_path = models_path
    else:
        return ''

    if ext is None:
        ext = data_file_ext

    if number is None:
        if target_subdir is not None and target_subdir !='':
            file_fullpath = os.path.join(root_path, target_subdir, f"{file_name}.{ext}")
        else:
            file_fullpath = os.path.join(root_path, f"{file_name}.{ext}")
    else:
        if target_subdir is not None and target_subdir !='':
            file_fullpath = os.path.join(root_path, target_subdir, f"{file_name}_{number:03}.{ext}")
        else:
            file_fullpath = os.path.join(root_path, f"{file_name}_{number:03}.{ext}")

    if is_compressed:
        file_fullpath = f'{file_fullpath}.{arch_ext}'

    return file_fullpath


def get_run_script_fullpath(file_name):
    """
    Конструктор полного имени запускаемого файла скрипта
    :param file_name: Название файла скрипта
    :return: Полный путь до файла
    """

    return os.path.join(all_scripts_path, run_scripts_dir, f"{file_name}.py")

# абсолютный путь проекта
project_path = settings.get_fresh('PROJECT_PATH')
if project_path is None:
    project_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))

# Путь до хранилища с данными
datastorage_path = settings.get_fresh('DATASTORAGE_PATH')

# Путь до каталога с данными
all_data_dir = settings.get_fresh('DATA_LOCAL_DIR', 'data')
all_data_path = os.path.join(datastorage_path, all_data_dir)

# Путь до каталога с моделями
models_dir = settings.get_fresh('MODEL_LOCAL_DIR', 'models')
models_path = os.path.join(datastorage_path, models_dir)

# Путь до каталога со скриптами
all_scripts_dir = settings.get_fresh('SCRIPT_DIR', 'scripts')
all_scripts_path = os.path.join(project_path, all_scripts_dir)
run_scripts_dir = settings.get_fresh('RUN_SCRIPT_SUBDIR', 'run')

# Расширения файлов
data_file_ext = settings.get_fresh('DATA_FILE_FORMAT', 'csv')
model_file_ext = 'txt'
arch_ext = settings.get_fresh('DATA_ARCH_FORMAT', 'gz')
