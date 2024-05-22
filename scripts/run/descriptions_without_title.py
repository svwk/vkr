#! python3
# -*- coding: UTF-8 -*-

# %% Импорт
import io
import gzip
import csv
from config import settings

import scripts.utils.files as files


subdir = settings.get_fresh('PROCESSED_LOCAL_SUBDIR')

data_file_blocks_name = 'blocks_extracted'
data_file_blocks = files.get_file_fullpath(data_file_blocks_name, subdir, is_compressed=True)
data_file_blocks_ed = files.get_file_fullpath(f'{data_file_blocks_name}_without_titles', subdir, is_compressed=False)

titles = {}
index = 0

print(f'Начинается обработка файла данных {data_file_blocks} ')

try:
    with gzip.open(data_file_blocks, 'rt', encoding='utf8', newline='\n') as f_in_gz:
        with io.open(data_file_blocks_ed, "w", encoding="utf8") as f_out:
            file_reader = csv.reader(f_in_gz, delimiter=",")
            file_writer = csv.writer(f_out, delimiter=",", lineterminator="\n")
            index = 0
            for line in file_reader:
                index += 1
                if index == 1 or line[1] == '':
                    file_writer.writerow(line)

except Exception as e:
    print(e)

print('Выполнение закончено')
