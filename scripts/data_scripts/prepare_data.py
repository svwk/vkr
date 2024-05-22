import pandas as pd
from bs4 import BeautifulSoup, element
import csv
from config import settings

from scripts.data_scripts import clear_text as cl


class ContentTypes:
    """
    Тип содержимого блока
    """
    # текстовый
    text = 0
    # список
    list = 1
    # возможно список
    list_br = 2
    #  только заголовок
    block_title = 3


class SemanticTypes:
    """
    Семантический тип
    """
    # Неизвестное назначение
    unknown = 0
    # описание вакансии
    description = 1
    # обязанности
    responsibilities = 2
    # требования
    requirements = 3
    # желательные требования
    desirable = 4
    # условия работы
    conditions = 5


def remove_unnamed_cols(dataframe):
    """
    Удаление безымянных столбцов, создаваемых при загрузке файла с ошибками
    :param dataframe: Датафрейм, в котором производятся действия
    :return: Количество удаленных столбцов
    """

    wrong_cols = [col for col in dataframe.columns if 'Unnamed' in col]

    dataframe.drop(dataframe.loc[dataframe[wrong_cols].notna().any(axis=1)].index, inplace=True)
    dataframe.drop(columns=wrong_cols, inplace=True)

    dataframe.reset_index(drop=True, inplace=True)

    return len(wrong_cols)


def drop_spec_cols(dataframe):
    """
    Удаление столбцов с информацией о специализации с сайта hh.ru
    :param dataframe: Датафрейм, в котором производятся действия
    :return: Количество удаленных столбцов
    """

    wrong_cols = [col for col in dataframe.columns if 'spec_' in col]
    dataframe.drop(columns=wrong_cols, inplace=True)

    return len(wrong_cols)


def fill_nans(dataframe):
    """
    Заполнение пропусков
    :param dataframe: Датафрейм, в котором производятся действия
    :param default_values: Словарь значений, которыми будут заполняться пропуски
    :return: Количество оставшихся пустых значений
    """
    default_values = settings.get_fresh('types.DEFAULT_VALUES')
    dataframe.fillna(value=default_values, inplace=True)
    return dataframe.isnull().sum()


def fix_data_types(dataframe):
    """
    Исправление типов данных признаков
    :param dataframe: Датафрейм, в котором производятся действия
    :return:
    """
    # Переведем признак published_at в тип данных 'datetime'
    dataframe['published_at'] = pd.to_datetime(dataframe['published_at'])

    # Переведем признак area_id в тип данных 'int'
    dataframe['area_id'] = pd.to_numeric(dataframe['area_id'], downcast='unsigned')
    # Переведем признак employer_id в тип данных 'int'
    dataframe['employer_id'] = pd.to_numeric(dataframe['employer_id'], downcast='unsigned')

    # Переведем признак salary_from в тип данных 'float'
    dataframe['salary_from'] = pd.to_numeric(dataframe['salary_from'], downcast='unsigned')
    # Переведем признак salary_to в тип данных 'float'
    dataframe['salary_to'] = pd.to_numeric(dataframe['salary_to'], downcast='unsigned')


def cat_to_num(dataframe):
    """
    Приведение признаков, имеющих категориальный характер, к целому типу
    :param dataframe: Датафрейм, в котором производятся действия
    """
    experience_id_mapper = settings.get_fresh('mappers.EXPERIENCE_ID')
    # schedule_id_mapper = settings.get_fresh('mappers.SCHEDULE_ID')
    salary_currency_mapper = settings.get_fresh('mappers.SALARY_CURRENCY')
    # employment_id_mapper = settings.get_fresh('mappers.EMPLOYMENT_ID')

    dataframe['experience_id'] = pd.to_numeric(dataframe['experience_id'].map(experience_id_mapper),
                                               downcast='unsigned')
    # dataframe['schedule_id'] = pd.to_numeric(dataframe['schedule_id'].map(schedule_id_mapper), downcast='unsigned')
    dataframe['salary_currency'] = pd.to_numeric(dataframe['salary_currency'].map(salary_currency_mapper),
                                                 downcast='unsigned')
    # dataframe['employment_id'] = pd.to_numeric(dataframe['employment_id'].map(employment_id_mapper),
    # downcast='unsigned')


def add_shortname_feature(dataframe):
    """
    Добавление нового признака - укороченное название short_name
    :param dataframe: Датафрейм, в котором производятся действия
    """
    dataframe['short_name'] = dataframe['name'].apply(lambda r: cl.preprocess_name(r))
    dataframe.drop(dataframe[dataframe['short_name'] == ''].index, inplace=True)


def drop_duplicates(df_to_search, df_to_drop=None):
    """
    Удаление дубликатов
    :param df_to_search: Датафрейм, в котором производится поиск дубликатов
    :param df_to_drop: Датафрейм, в котором производится удаление дубликатов
    :return: Количество удаленных записей
    """
    if df_to_drop is None and df_to_search is None:
        return 0
    elif df_to_drop is None:
        df_to_drop = df_to_search
    elif df_to_search is None:
        df_to_search = df_to_drop

    dropped_count = 0
    same_cols = ['description', 'employer_id']
    df_with_duplicates = df_to_search[df_to_search[same_cols].duplicated(keep=False)]

    dropped_count += drop_shortname_publishedat_duplicates(df_with_duplicates, df_to_drop)
    dropped_count += drop_shortname_area_duplicates(df_with_duplicates, df_to_drop)
    dropped_count += drop_shortname_keyskills_duplicates(df_with_duplicates, df_to_drop)

    df_to_drop.reset_index(drop=True, inplace=True)

    return dropped_count


def _drop_by_id(id_list, df_to_search, df_to_drop=None):
    """
    Удаление строк в датафреймах по идентификаторам
    :param df_to_search: Датафрейм, в котором производится поиск дубликатов
    :param df_to_drop: Датафрейм, в котором производится удаление дубликатов
     :param id_list: Список идентификаторов для удаления
    :return: Количество удаленных записей
    """
    if df_to_drop is None and df_to_search is None or id_list is None or len(id_list) == 0:
        return 0
    elif df_to_drop is None:
        df_to_drop = df_to_search
    elif df_to_search is None:
        df_to_search = df_to_drop

    df_to_drop.drop(df_to_drop[df_to_drop.id.isin(id_list)].index, inplace=True)
    if df_to_search is not df_to_drop:
        df_to_search.drop(df_to_search[df_to_search.id.isin(id_list)].index, inplace=True)
    return len(id_list)


def drop_all_fields_duplicates(df_to_search, df_to_drop=None, to_reset_index=False):
    """
    Удаление полных дубликатов (по пяти полям сразу)
    :param to_reset_index: Флаг для перестройки индекса после удаления дубликатов
    :param df_to_search: Датафрейм, в котором производится поиск дубликатов
    :param df_to_drop: Датафрейм, в котором производится удаление дубликатов
    :return: Количество удаленных записей
    """
    if df_to_drop is None and df_to_search is None:
        return 0
    elif df_to_search is None:
        df_to_search = df_to_drop

    same_cols = ['name', 'description', 'experience_id', 'key_skills', 'employer_id']
    duplicate_flags = df_to_search[same_cols].duplicated(keep='last')
    df_with_duplicates = df_to_search[duplicate_flags]
    ids = df_with_duplicates.id.values
    dropped_count = _drop_by_id(ids, df_to_search, df_to_drop)

    if to_reset_index:
        if df_to_drop is not None:
            df_to_drop.reset_index(drop=True, inplace=True)
        if df_to_search is not df_to_drop:
            df_to_search.reset_index(drop=True, inplace=True)

    return dropped_count


def drop_shortname_publishedat_duplicates(df_to_search, df_to_drop=None, to_reset_index=False):
    """
    Удаление дубликатов по полям 'название' и 'дата публикации'
    :param to_reset_index: Флаг для перестройки индекса после удаления дубликатов
    :param df_to_search: Датафрейм, в котором производится поиск дубликатов
    :param df_to_drop: Датафрейм, в котором производится удаление дубликатов
    :return: Количество удаленных записей
    """
    # Удаляем: одинаковые описания, названия, работодатель, дата публикации.
    # Из дублей оставляем тот вариант, у которого навыки длиннее.
    dropped_count = 0
    if df_to_drop is None and df_to_search is None:
        return 0
    elif df_to_search is None:
        df_to_search = df_to_drop

    same_cols = ['short_name', 'published_at']
    duplicate_flags = df_to_search[same_cols].duplicated(keep=False)
    df_with_duplicates = df_to_search[duplicate_flags].groupby(same_cols)

    for name, group in df_with_duplicates:
        new_df_to_search = group.sort_values(by='key_skills', key=lambda col: col.str.len(), ascending=False)
        ids = new_df_to_search.id.values[1:]
        dropped_count += _drop_by_id(ids, new_df_to_search, df_to_drop)

    if to_reset_index:
        if df_to_drop is not None:
            df_to_drop.reset_index(drop=True, inplace=True)
        if df_to_search is not df_to_drop:
            df_to_search.reset_index(drop=True, inplace=True)

    return dropped_count


def drop_shortname_area_duplicates(df_to_search, df_to_drop=None, to_reset_index=False):
    """
    Удаление дубликатов по полям 'название' и 'регион'
    :param to_reset_index: Флаг для перестройки индекса после удаления дубликатов
    :param df_to_search: Датафрейм, в котором производится поиск дубликатов
    :param df_to_drop: Датафрейм, в котором производится удаление дубликатов
    :return: Количество удаленных записей
    """
    # Удаляем: одинаковые описания, названия, работодатель, регион.
    # Из дублей оставляем тот вариант, у которого навыки длиннее.
    dropped_count = 0
    if df_to_drop is None and df_to_search is None:
        return 0
    elif df_to_search is None:
        df_to_search = df_to_drop

    same_cols = ['short_name', 'area_id']
    duplicate_flags = df_to_search[same_cols].duplicated(keep=False)
    df_with_duplicates = df_to_search[duplicate_flags].groupby(same_cols)

    for name, group in df_with_duplicates:
        new_df_to_search = group.sort_values(by='key_skills', key=lambda col: col.str.len(), ascending=False)
        ids = new_df_to_search.id.values[1:]
        dropped_count += _drop_by_id(ids, new_df_to_search, df_to_drop)

    if to_reset_index:
        if df_to_drop is not None:
            df_to_drop.reset_index(drop=True, inplace=True)
        if df_to_search is not df_to_drop:
            df_to_search.reset_index(drop=True, inplace=True)

    return dropped_count


def drop_shortname_keyskills_duplicates(df_to_search, df_to_drop=None, to_reset_index=False):
    """
    Удаление дубликатов по полям 'название' и 'навыки'
    :param to_reset_index: Флаг для перестройки индекса после удаления дубликатов
    :param df_to_search: Датафрейм, в котором производится поиск дубликатов
    :param df_to_drop: Датафрейм, в котором производится удаление дубликатов
    :return: Количество удаленных записей
    """
    # Удаляем: одинаковые описания, названия, работодатель, навыки.
    # Из дублей оставляем вариант с наименьшим опытом.
    dropped_count = 0
    if df_to_drop is None and df_to_search is None:
        return 0
    elif df_to_search is None:
        df_to_search = df_to_drop

    same_cols = ['short_name', 'key_skills']
    duplicate_flags = df_to_search[same_cols].duplicated(keep=False)
    df_with_duplicates = df_to_search[duplicate_flags].groupby(same_cols)

    for name, group in df_with_duplicates:
        new_df_to_search = group.sort_values(by='experience_id', ascending=True)
        ids = new_df_to_search.id.values[1:]
        dropped_count += _drop_by_id(ids, new_df_to_search, df_to_drop)

    if to_reset_index:
        if df_to_drop is not None:
            df_to_drop.reset_index(drop=True, inplace=True)
        if df_to_search is not df_to_drop:
            df_to_search.reset_index(drop=True, inplace=True)

    return dropped_count


def concat_dataframes(dataframe_list):
    """
    Соединение нескольких датафреймов в один
    :param dataframe_list: Список датафреймов для соединения
    :return: Новый датафрейм
    """
    return pd.concat(dataframe_list, axis=0, ignore_index=True)


def prepare_data_for_parse(dataframe):
    '''

    :param dataframe:  Датафрейм после первого этапа обработки
    :return:
    '''

    # Оставим в датасете только строки, для которых salary_currency=0 (RUR)
    # и в которых поле 'key_skills' (Ключевые навыки) заполнено :
    dataframe = dataframe[(dataframe.salary_currency == 0) & (dataframe.key_skills.notna())]

    # Удалим ненужные для обучения поля
    wrong_cols = ['experience_id', 'published_at', 'employer_id', 'salary_from', 'salary_to',
                  'salary_currency', 'area_id', 'area_name']
    dataframe.drop(columns=wrong_cols, inplace=True, errors='ignore')

    return dataframe


def parse_description(id_hh, text, target_dict, start_index):
    if text is None or text == '':
        return None

    soup = BeautifulSoup(text, 'lxml')
    if soup is None or len(soup.contents) == 0:
        return None

    index = None
    old_title = ''
    old_content = ''
    old_index = start_index
    content_type = ContentTypes.text
    block_id = 1

    for item in soup.html.body.contents:
        if len(item.text) <= 2:
            continue

        # Заголовок
        if (item.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'em', 'b', 'strong'] and len(item.text) < 60 or
                item.name == 'p' and 1 < len(item.text) < 50 and
                # если это текст, а  после - списки, то это - заголовок
                (item.next_sibling is not None and item.next_sibling.name in ['ul', 'ol'] or item.text[-1] == ':')):

            index = old_index + 1
            title = item.text.strip()
            content = ''
            content_type = ContentTypes.block_title

        elif item.name in ['ul', 'ol']:
            # Если это список
            content = '\n'.join([li.text for li in item.contents if len(li.text) > 0])

            if old_index > 0 and content_type == ContentTypes.block_title and old_content == '':
                title = old_title
                index = old_index
            elif old_index > 0 and content_type == ContentTypes.list:
                content = f'{old_content}\n{content}'
                title = old_title
                index = old_index
            else:
                title = ''
                index = old_index + 1

            content_type = ContentTypes.list

        elif isinstance(item, element.Tag) and item.find('br') is not None and item.contents is not None:
            children = item.contents
            if len(children[0].text) > 1 and children[0].text[-1] == ':':
                title = children[0].text
                children = item.contents[1:]
                index = old_index + 1
            elif old_index > 0 and content_type == ContentTypes.block_title and old_content == '':
                title = old_title
                index = old_index
            else:
                title = ''
                index = old_index + 1

            content = '\n'.join([el.text for el in children if el.name != 'br' and len(el.text) > 1])
            content_type = ContentTypes.list_br

        else:
            # Обычный текстовый блок; если до этого тоже был текстовый блок, то объединяем
            content = item.text
            if old_index > 0 and content_type == ContentTypes.block_title and old_content == '':
                title = old_title
                index = old_index
            elif old_index > 0 and content_type == ContentTypes.text:
                content = f'{old_content}\n{content}'
                title = old_title
                index = old_index
            else:
                title = ''
                index = old_index + 1

            content_type = ContentTypes.text

        if title is not None and len(title) > 0:
            title = cl.remove_description_postfix(title)
            title = cl.remove_name_prefix(title)
            title = cl.remove_new_lines(title)

        if content is not None and len(content) > 0:
            content = cl.remove_description_postfix(content)
            content = cl.remove_name_prefix(content)

        if len(content) > 0 or len(title) > 0:
            target_dict[index] = {'id': id_hh, 'title': title,
                                        'content': content,
                                        'content_type': content_type,
                                        'semantic_type': SemanticTypes.unknown,
                                        'block_id': block_id}

            # если предыдущий блок - заголовок, а мы создаем новый заголовок,
            #  то предыдущий переводится в статус текстового блока
            if (old_index in target_dict.keys() and
                    target_dict[old_index]['content_type'] == ContentTypes.block_title and old_content == ''):
                target_dict[old_index]['content_type'] = ContentTypes.text
                target_dict[old_index]['content'] = target_dict[old_index]['title']
                target_dict[old_index]['title'] = ''

            block_id += 1
            old_index = index
            old_content = content
            old_title = title

    return index


def parse_all_descriptions(dataframe):
    # new_df = pd.DataFrame({'id': [], 'title': [], 'content': [], 'content_type': [], 'semantic_type': []})
    # the dictionary to pass to pandas dataframe
    dict = {}
    index = 0

    for row in list(zip(*dataframe.to_dict("list").values())):
        item = cl.preprocess_description(row[2])
        index = parse_description(row[0], item, dict, index)

    new_df = pd.DataFrame.from_dict(dict, "index")
    new_df['id'] = pd.to_numeric(new_df['id'], downcast='unsigned')
    new_df['content_type'] = pd.to_numeric(new_df['content_type'], downcast='unsigned')
    new_df['semantic_type'] = pd.to_numeric(new_df['semantic_type'], downcast='unsigned')

    return new_df


def update_description_semantic_type(f_in, f_out, titles_dict):
    file_reader = csv.reader(f_in, delimiter=",")
    file_writer = csv.writer(f_out, delimiter=",", lineterminator="\n")
    index = 0
    for line in file_reader:
        index += 1
        if index == 1:
            file_writer.writerow(line)
            continue

        title = line[1]
        semantic_type = titles_dict.get(title, line[4])
        semantic_type = line[4] if semantic_type == '' else semantic_type
        # line[6] = '1' if semantic_type != '0' and semantic_type != '' else line[6]
        line[4] = semantic_type
        # if line[6] == '1':
        file_writer.writerow(line)
