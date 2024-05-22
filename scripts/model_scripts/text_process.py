import pandas as pd
import os
import re
import pymorphy3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from config import settings
from scripts.data_scripts import clear_text as cl
import scripts.utils.files as files


model_subdir = settings.get_fresh('BLOCK_MODELING_SUBDIR')
processed_subdir = settings.get_fresh('PROCESSED_LOCAL_SUBDIR')
nltk_dir = settings.get_fresh('NLTK_DIR')
punctuation_marks =  settings.get_fresh('PUNCTUATION_MARKS')

# nltk.download('punkt', download_dir=nltk_dir)
# nltk.download('stopwords', download_dir=nltk_dir)

stopwords_add = ['т.д', 'т.п', 'др', 'пр']
stopwords = (set(stopwords.words('russian'))
              .union(set(stopwords.words('english')))
              .union(set(stopwords_add)))

morph = pymorphy3.MorphAnalyzer()

def parse_all_key_skills(dataframe):
    '''
    Парсинг поля key_skills в датасете 'preprocessed'.
    Формирует новый датасет, содержащий элементы списка из столбца
    с номером 3
    :param dataframe: Датасет, в котором столбец с номером 0
    содержит идентификатор объявления на hh.ru, а столбец с номером 3
    надо распарсить в список. Этот столбец содержит данные списка, соединенные
    символом перехода строки '\n'
    :return: новый датасет, строками которого являются распарсенные
    элементы списка из столбца с номером 3
    '''
    # return parse_list_items(dataframe, 3)
    result_dict = {}
    index = 0
    for row in list(zip(*dataframe.to_dict("list").values())):
        items = row[3].split('\n')
        for item in items:
            if item in punctuation_marks:
                continue
            index += 1
            item = cl.preprocess_listitem(item)
            result_dict[index] = {'id': row[0], 'key_skill': item}

    new_df = pd.DataFrame.from_dict(result_dict, "index")
    new_df['id'] = pd.to_numeric(new_df['id'], downcast='unsigned')

    return new_df

def parse_requirements(dataframe):
    '''
    Парсинг поля content в датасете 'FILENAME_BLOCKS_EXTRACTED'.
    Формирует новый датасет, содержащий элементы списка из столбца
    с номером 2
    :param dataframe: Датасет, в котором столбец с номером 0
    содержит идентификатор объявления на hh.ru, а столбец с номером 2
    надо распарсить в список. Этот столбец содержит данные списка, соединенные
    символом перехода строки '\n'
    :return: новый датасет, строками которого являются распарсенные
    элементы списка из столбца с номером 2
    '''
    result_dict = {}
    index = 0
    stop_words_for_list = ['if']
    for row in list(zip(*dataframe.to_dict("list").values())):
        if (isinstance(row[2], str) is False or row[2] == '' or
                isinstance(row[1], str) is False or row[1] == '') :
            continue

        has_stop_words = [i for i in stop_words_for_list if i in row[2]]
        if len(has_stop_words) > 0:
            continue

        items = cl.divide_to_list(row[2])
        items = items.split('\n')
        for item in items:
            if item in punctuation_marks:
                continue
            index += 1
            item = cl.preprocess_listitem(item)
            if item is not None and len(item) > 1:
                result_dict[index] ={'id': row[0],
                                   'block_id': row[5],
                                   'semantic_type': row[4],
                                   'list_item': item}

    new_df = pd.DataFrame.from_dict(result_dict, "index")
    new_df['id'] = pd.to_numeric(new_df['id'], downcast='unsigned')
    new_df.drop(new_df[new_df.list_item.isna()].index, inplace=True)

    return new_df


def tokenize_requirements_text(processed_text):
    if processed_text is None or not isinstance(processed_text, str) or len(processed_text) < 2:
        return None

    tokens = word_tokenize(processed_text.lower())
    token_list=[]
    bigram_list=[]
    trigram_list=[]
    token_old = None
    token_old_old = None
    # tokens = gensim.utils.simple_preprocess(processed_text)

    for token in tokens:
        if token in punctuation_marks:
            continue
        lemma = morph.parse(token)[0].normal_form
        # lemma = token
        if not lemma in stopwords:
            token_list.append(lemma)
            if token_old is not None:
                bigram_list.append((token_old, lemma))
            if token_old is not None and token_old_old is not None:
                trigram_list.append((token_old_old, token_old, lemma))
            token_old_old = token_old
            token_old = lemma

    return {'unigrams': token_list,
            'bigrams': bigram_list,
            'trigrams': trigram_list}


def tokenize_requirement_dataset(dataframe):
    pid = os.getpid()
    print(f'Процесс {pid} начал свою работу')
    dataframe.drop(dataframe[dataframe.list_item.isna()].index, inplace=True)
    # dataframe.drop(dataframe[dataframe['list_item'].duplicated(keep='last')].index, inplace=True)

    token_dict = {}
    for row in list(zip(*dataframe.to_dict("list").values())):
        processed_text = row[3]
        if (processed_text is None or
                not isinstance(processed_text, str) or
                len(processed_text) < 2
                or not cl.has_word_symbols(processed_text)):
            continue

        tokenized_values = tokenize_requirements_text(row[3])
        if (tokenized_values is not None and tokenized_values['unigrams'] is not None
                and len(tokenized_values['unigrams']) > 0) :
            key = '_'.join(tokenized_values['unigrams'])
            if key in token_dict:
                token_dict[key]['count'] += 1
                continue
            token_dict[key] = {'count': 1,
                               'canonical': row[3],
                               'unigrams': tokenized_values['unigrams'],
                               'bigrams': tokenized_values['bigrams'],
                               'trigrams': tokenized_values['trigrams']}
    print(f'Процесс {pid} закончил свою работу')
    return token_dict


def concat_token_dict(token_dict_list):
    new_dict = token_dict_list[0].copy()
    for token_dict in token_dict_list[1:]:
        for key in token_dict:
            if key in new_dict:
                new_dict[key]['count'] += token_dict[key]['count']
                continue
            else:
                new_dict[key] = token_dict[key]

    return new_dict


def tokenize_blocks(dataframe):
    '''
    Подготовка поля content в датасете 'FILENAME_BLOCKS_EXTRACTED' к векторизации.
    :param dataframe: Датасет, в котором столбец с номером 0
    содержит  заголовок блока, а столбец с номером 1
    надо обработать. Этот столбец содержит данные списка, соединенные
    символом перехода строки '\n'
    :return: Новый датасет, содержащий новый столбец, состоящий из заголовка и содержимого блока, готовый к векторизации
    '''
    # new_df = pd.DataFrame({'id': [], 'title': [], 'content': [], 'content_type': [], 'semantic_type': []})
    # the dictionary to pass to pandas dataframe
    pid = os.getpid()
    print(f'Процесс {pid} начал свою работу')

    tokenized_blocks = []
    _pattern_punkt_characters_to_whitespace = re.compile(r'[:,;-]+')
    _pattern_punkt_characters_to_lineend = re.compile(r'[?!]+')

    for row in list(zip(*dataframe.to_dict("list").values())):
        content = row[1]
        if (content is None or
                not isinstance(content, str) or
                len(content) < 2 or
                not cl.has_word_symbols(content)):
            tokenized_blocks.append('')
            continue

        content = cl.remove_new_lines(content, '')
        content = content.replace('\n', '.')
        title = row[0]
        if title is not None and isinstance(title, str) and len(title) > 1 and cl.has_word_symbols(title):
            content = f'{title}.{content}'
        content=_pattern_punkt_characters_to_whitespace.sub(' ', content)
        content = _pattern_punkt_characters_to_lineend.sub('.', content)
        sentences = content.split('.')
        new_sentences = []
        for sent in sentences:
            token_list = []
            if sent in punctuation_marks:
                continue
            sent = sent.strip().lower()
            sent = cl.preprocess_listitem(sent)
            tokens = sent.split()
            for token in tokens:
                if token in punctuation_marks:
                    token_list.append(token)
                    continue
                lemma = morph.parse(token)[0].normal_form
                # lemma = token
                token_list.append(lemma)

            new_sentences.append(' '.join(token_list))

        tokenized_blocks.append('.'.join(new_sentences))

    if len(tokenized_blocks) == dataframe.shape[0]:
        dataframe['tokenized_block'] = tokenized_blocks

    print(f'Процесс {pid} закончил свою работу')
    return dataframe


def join_unigrams(unigrams):
    if unigrams is None or len(unigrams) == 0:
        return ''
    return ' '.join(unigrams)



