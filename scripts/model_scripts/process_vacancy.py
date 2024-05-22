#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
import torch
import warnings
from config import settings
import argparse
from pprint import pprint

import scripts.utils.files as files
import scripts.model_scripts.cluster_utils as cu
from scripts.data_scripts import clear_text as cl
import scripts.data_scripts.prepare_data as prep
import scripts.model_scripts.text_process as proc

# Настройка параметров
warnings.simplefilter("ignore")

RANDOM_STATE = 123
MODEL_SUBDIR = settings.get_fresh('BLOCK_MODELING_SUBDIR')
PROCESSED_SUBDIR = settings.get_fresh('PROCESSED_LOCAL_SUBDIR')
READY_DATA_LOCAL_SUBDIR = settings.get_fresh('READY_DATA_LOCAL_SUBDIR')

FILENAME_TITLES_EXTRACTED_ED = settings.get_fresh('FILENAME_TITLES_EXTRACTED_ED')
FILENAME_BLOCK_VECTORIZATION_MODEL = settings.get_fresh('FILENAME_BLOCK_VECTORIZATION_MODEL')
FILENAME_BLOCK_CLASSIFICATION_MODEL = settings.get_fresh('FILENAME_BLOCK_CLASSIFICATION_MODEL')
FILENAME_REQUIREMENT_VECTORIZATION_MODEL = settings.get_fresh('FILENAME_REQUIREMENT_VECTORIZATION_MODEL')
FILENAME_BLOCK_REDUCER_MODEL = settings.get_fresh('FILENAME_BLOCK_REDUCER_MODEL')

FILENAME_ALL_KEYWORD = settings.get_fresh('FILENAME_ALL_KEYWORD')
FILENAME_CLUSTER_KEYWORDS = settings.get_fresh('FILENAME_CLUSTER_KEYWORDS')
FILENAME_CLUSTER_DATA = settings.get_fresh('FILENAME_CLUSTER_DATA')

CLUSTERIZATION_MODEL_NAME = 'paraphrase_minilm_l12'
CLASSIFICATION_MODEL_NAME = 'paraphrase_mpnet_v2'

MODELS = {
    CLUSTERIZATION_MODEL_NAME:
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',

    CLASSIFICATION_MODEL_NAME:
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_vacancy(text):
    '''
    Извлечение текстовых блоков из описания вакансии
    :param text: Текст вакансии
    :return:
    '''

    if not isinstance(text, str):
        return {'Ошибка': f"Неверный тип данных, {type(text)}"}

    if text is None or len(text) < 2:
        return {'Ошибка': "Передан пустой текст"}

    # 2. Загрузка данных и моделей

    block_vectorization_filename = f'{CLASSIFICATION_MODEL_NAME}_{FILENAME_BLOCK_VECTORIZATION_MODEL}'
    block_vectorization_model = files.load_model(block_vectorization_filename)
    block_classification_model = files.load_model(FILENAME_BLOCK_CLASSIFICATION_MODEL)

    filename = settings.get_fresh('FILENAME_ALL_KEYWORD')
    all_keywords = files.load_file(filename, READY_DATA_LOCAL_SUBDIR, to_decompress=False, with_dates=False)

    filename = settings.get_fresh('FILENAME_CLUSTER_KEYWORDS')
    keywords = files.load_data_dump(filename, READY_DATA_LOCAL_SUBDIR)

    filename = settings.get_fresh('FILENAME_CLUSTER_DATA')
    cluster_data = files.load_file(filename, READY_DATA_LOCAL_SUBDIR, to_decompress=False, with_dates=False)

    # 4.Обработка текста вакансии
    # 4.1. Извлечение текстовых блоков из описания вакансии
    item = cl.preprocess_description(text)
    content_dict = {}
    start_index = 0
    index = prep.parse_description(0, item, content_dict, start_index)

    df = pd.DataFrame.from_dict(content_dict, "index")
    df['id'] = pd.to_numeric(df['id'], downcast='unsigned')
    df['content_type'] = pd.to_numeric(df['content_type'], downcast='unsigned')
    df['semantic_type'] = pd.to_numeric(df['semantic_type'], downcast='unsigned')
    df.reset_index(drop=True, inplace=True, )

    # 4.2. Классификация текстовых блоков объявлений
    # 4.2.1 Подготовка данных к классификации
    mapper = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2}
    df_cl = df.copy()
    df_cl['target_type'] = pd.to_numeric(df_cl['semantic_type'].map(mapper), downcast='unsigned')
    df_cl.drop(columns=['semantic_type', 'content_type', 'id'], inplace=True, errors='ignore')
    proc.tokenize_blocks(df_cl)

    df_cl['raw_block'] = df_cl.title
    df_cl.raw_block.fillna("", inplace=True)
    df_cl['raw_block'] = df_cl.raw_block.str.lower() + '. ' + df_cl.content.str.lower()
    df_cl.drop(columns=['title', 'content'], inplace=True, errors='ignore')

    # 4.2.2. Векторизация текста
    embeddings = block_vectorization_model.encode(df_cl.tokenized_block, device=DEVICE)
    df_cl['target_type'] = block_classification_model.predict(embeddings)

    # Блоки, имеющие класс 1, будут использоваться для извлечения требований:
    indexes = df_cl[df_cl.target_type == 1].index.tolist()
    df_selected = df.loc[indexes, :]

    # Проставляем семантический тип блока в исходный словарь
    for key in content_dict:
        block_id = content_dict[key]['block_id']
        if len(df_cl[df_cl.block_id == block_id]) > 0:
            content_dict[key]['semantic_type'] = list(df_cl[df_cl.block_id == block_id].target_type)[0]

    # 4.3. Извлечение требований
    list_items = proc.parse_requirements(df_selected)
    requirements_dict = proc.tokenize_requirement_dataset(list_items)

    # 4.4. Уточнение требований
    unigrams_counter, bigrams_counter, trigrams_counter = cu.get_all_skills(requirements_dict)

    # Присвоение меток классам, которые будут использоваться для отображения в вакансии
    labels = {}
    for cluster_key in keywords:
        if len(keywords[cluster_key].most_common()) > 1:
            key1 = keywords[cluster_key].most_common()[0][0]
            key2 = keywords[cluster_key].most_common()[1][0]
            if len(key2) > 0 and len(key2) >= len(key1) and key1 in key2:
                labels[cluster_key] = key2
            elif len(key1) > 0 and len(key1) > len(key2) and key2 in key1:
                labels[cluster_key] = key1
            elif len(key1) > 0:
                labels[cluster_key] = key1
        if len(keywords[cluster_key].most_common()) == 1:
            labels[cluster_key] = keywords[cluster_key].most_common()[0][0]

    # Определение основных и дополнительных навыков
    key_skills = set()
    add_skills = set()
    keyword_list = all_keywords.iloc[:, 0].tolist()

    for keyword_pair in unigrams_counter.most_common():
        keyword = keyword_pair[0]
        for cluster_key in list(keywords.keys()):
            if any(keyword in item for item in keywords[cluster_key].keys()):
                label = labels.get(cluster_key)
                if not None:
                    key_skills.add(label)
                break

        if keyword in keyword_list:
            add_skills.add(keyword)

    add_skills = add_skills - key_skills

    result = {}

    result['key_skills'] = key_skills
    result['add_skills'] = add_skills
    result['content'] = content_dict

    return result


if __name__ == "__main__":
    data_file = 'vacancy.html'
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Имя файла с описанием вакансии для обработки')
    namespace = parser.parse_args()
    if namespace.file:
        data_file = namespace.file

    if data_file is not None and  data_file != '':
        with open(data_file, 'r') as file:
            content = file.read()
            response = process_vacancy(content)
            pprint(response)

