#! python3
# -*- coding: UTF-8 -*-

# %% Импорт
import io
import gzip
import csv
from config import settings

import scripts.data_scripts.prepare_data as prep
import scripts.utils.files as files


subdir = settings.get_fresh('PROCESSED_LOCAL_SUBDIR')
data_file_titles_name = settings.get_fresh('FILENAME_TITLES_EXTRACTED_ED')
data_file_blocks_name = settings.get_fresh('FILENAME_BLOCKS_EXTRACTED')
data_file_blocks_ed_name = settings.get_fresh('FILENAME_BLOCKS_EXTRACTED_EDITED')
data_file_titles = files.get_file_fullpath(data_file_titles_name, subdir, is_compressed=False)
data_file_blocks = files.get_file_fullpath(data_file_blocks_name, subdir, is_compressed=True)
data_file_blocks_ed = files.get_file_fullpath(data_file_blocks_ed_name, subdir, is_compressed=True)

titles = {}
index = 0

print(f'Начинается загрузка файла данных {data_file_titles}')
with io.open(data_file_titles, encoding="utf8") as fd_in:
    file_reader = csv.reader(fd_in, delimiter=",")
    for line in file_reader:
        index += 1
        if index == 1:
            continue
        titles[line[0]] = line[2]

print(f'Начинается обработка файла данных {data_file_blocks} ')

try:
    with gzip.open(data_file_blocks, 'rt', encoding='utf8', newline='\n') as f_in_gz:
        with gzip.open(data_file_blocks_ed, 'wt') as f_out_gz:
            prep.update_description_semantic_type(f_in_gz, f_out_gz, titles)
except:
    with io.open(data_file_blocks, encoding="utf8") as fd_in:
        with gzip.open(data_file_blocks_ed, 'wt') as f_out_gz:
            prep.update_description_semantic_type(fd_in, f_out_gz, titles)

print('Выполнение закончено')
