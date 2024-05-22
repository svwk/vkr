import re
from importlib import reload

import config

reload(config)
from config import settings

stop_words_for_name = settings.get_fresh('STOP_WORDS_FOR_NAME')

_pattern_emoji = re.compile("["
                            u"\U0001F600-\U0001F64F"
                            u"\U0001F300-\U0001F5FF"
                            u"\U0001F680-\U0001F6FF"
                            u"\U0001F1E0-\U0001F1FF"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
_pattern_spec_symbols = re.compile(r'\S*(@|#)\S+')
_pattern_remove_numbers = re.compile(r'\s\d+\s')
_pattern_only_numbers = re.compile(r'^[^a-zA-Zа-яА-Я]+$')
_pattern_noword_characters = re.compile(r'[^a-zA-Zа-яА-Я0-9\s?]')
_pattern_russian_characters = re.compile(r'[а-яА-Я]{4,}')
_pattern_any_russian_characters = re.compile(r'[а-яА-Я]+')
_pattern_list_characters = re.compile(r'[^a-zA-Zа-яА-Я0-9\s+#-.*]+')

_pattern_punkt_to_whitespace_characters = re.compile(r'(?:[:()\[\]_]+|\s-|-\s)')
_pattern_url_characters = re.compile(r'((https|http)?:\/\/|www\.)(\w|\.|\/|\?|\=|\&|\%|-)*')
_pattern_whitespaces = re.compile(r'\s{2,}')
_pattern_double_symbols = re.compile(r'([^+a-zA-Zа-яА-Я0-9])\1+')
_pattern_tail = re.compile(r'[\(\[;\|\<!].*$')
_pattern_description_prefix = re.compile(r'^\s*[^a-zA-Zа-яА-Я0-9<>]+')
_pattern_description_postfix = re.compile(r'\s*[^a-zA-Zа-яА-Я0-9<>]+?$')
# Все небуквенные символы в начале строки, кроме . и *
_pattern_name_prefix1 = re.compile(r'^\s*[^a-zA-Zа-яА-Я0-9.*]+')
# Символы . или * в начале строки, кроме тех, которые перед названиями
_pattern_name_prefix3 = re.compile(r'^\s*[.*]+(?=[^a-zA-Z])[^a-zA-Zа-яА-Я0-9]*')
# Многократные . и *
_pattern_name_prefix2 = re.compile(r'^\s*([.*]){2,}')
_pattern_name_postfix = re.compile(r'\s*[^a-zA-Zа-яА-Я0-9+#]+?$')
_pattern_between_square_brackets = re.compile('\[[^]]*\]?')
_pattern__between_parentheses = re.compile('\([^)]*\)')
_pattern_inside_tag_whitespaces = re.compile(r'(\s+)(?=>)|(?<=</)(\s+)|(?<=<)(\s+)')
_pattern_between_tag_whitespaces = re.compile(r'(?<=>)\s+(?:\\n)*\s*(?=<)')
_pattern_between_par_tag_whitespaces = (
    re.compile(r'(\s*)(<p>|</p>|<ul>|</ul>|<ol>|</ol>|<li>|</li>|<h1>|</h1>)(\s*)'))
_pattern_empty_tags = re.compile(r'(<(\w+)[^>]*>\s*</\2>)')
_pattern_triple_tags = re.compile(r'(<(\w+)[^>]*>)\s*(<\2[^>]*>)\s*(<\2[^>]*>)([^<]*)(</\s?\2>)\s*(</\s?\2>)\s*(</\s?\2>)')
_pattern_double_tags = re.compile(r'(<(\w+)[^>]*>)\s*(<\2[^>]*>)([^<]*)(</\2>)\s*(</\2>)')
_pattern_pstrong_tags = re.compile(r'(?:<(p|em)[^>]*>\s*<strong>)([^<]*)(?:</\s?strong>\s*</\s?(p|em)[^>]*>)')
_pattern_h_tags = re.compile(r'h[2-5]')
_pattern_new_lines = re.compile(r'^(\s*\\n)+')
_pattern_pbr_open_tags = re.compile(r'(<p>\s*<br\s*/>)|(<br\s*/>\s*<p>)')
_pattern_pbr_close_tags = re.compile(r'(</p>\s*<br\s*/>)|(<br\s*/>\s*</p>)')
_pattern_list_dividers = re.compile(r'(?:[;|\/\\!?:]+|\d\)|(?:\s+и\s+)|(?:(?<=[^+])\+(?=[^+]))|(?:\.+\s)|(?:\,+(?=[^0-9])))+')

def has_any_russian_symbol(processed_text):
    """
    Does text contain at least 42 russian symbols?
    """
    return _pattern_any_russian_characters.search(processed_text) is not None

def has_russian_symbols(processed_text):
    """
    Does text contain at least 42 russian symbols?
    """
    return _pattern_russian_characters.search(processed_text) is not None

def has_word_symbols(processed_text):
    """
    Does text contain at least 2 russian symbols?
    """
    return _pattern_only_numbers.search(processed_text) is None

def remove_url(processed_text):
    """
    Удаление в тексте эмодзи
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    return _pattern_url_characters.sub('', processed_text)

def remove_emails(text):
    return re.sub(r"\S*@\w*\.\w*(\s?|,|\.)",'',text)

def remove_bots(text):
    return re.sub(r"\@[\w\d]*",'',text)


def clean_chars_with_regex(processed_text):
    """
    Удаление в тексте различных символов с помощью регулярных выражений
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    pattern = re.compile(r'[â]')
    clean_string = re.sub(pattern, '', processed_text)
    return clean_string


def remove_emoji(processed_text):
    """
    Удаление в тексте эмодзи
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    return _pattern_emoji.sub(r'', processed_text)


def remove_numbers(processed_text):
    """
    Удаление в тексте чисел
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_remove_numbers.sub('', processed_text)
    return processed_text


def remove_spec_symbols(processed_text):
    """
    Удаление в тексте специальных символов
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_spec_symbols.sub('', processed_text)
    return processed_text


def remove_nolist_characters(processed_text):
    """
    Удаление в тексте специальных символов
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_list_characters.sub('', processed_text)
    return processed_text

def remove_double_symbols(processed_text):
    """
    Удаление в тексте многократных повторений символов
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_double_symbols.sub(' ', processed_text)
    processed_text = processed_text.strip()
    return processed_text

def remove_whitespaces(processed_text):
    """
    Удаление в тексте лишних пробелов
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_whitespaces.sub(' ', processed_text)
    processed_text = processed_text.strip()
    return processed_text

def remove_punkt_symbols(processed_text):
    """
    Замена некоторых символов в тексте на пробелы
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_punkt_to_whitespace_characters.sub(' ', processed_text)
    processed_text = processed_text.strip()
    return processed_text



def remove_tag_whitespaces(processed_text):
    """
     Удаление в тексте лишних пробелов вокруг тегов
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_inside_tag_whitespaces.sub('', processed_text)
    processed_text = _pattern_inside_tag_whitespaces.sub('', processed_text)
    processed_text = _pattern_between_par_tag_whitespaces.sub(lambda m: m.group(2), processed_text)
    return processed_text


def remove_empty_tags(processed_text):
    """
    Удаление пустых тегов
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_empty_tags.sub('', processed_text)
    return processed_text

def remove_duplicate_tags(processed_text):
    """
    Удаление двойных и тройных вложенных тегов
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_triple_tags.sub(lambda m: f'<{m.group(2)}>{m.group(5)}</{m.group(2)}>', processed_text)
    processed_text = _pattern_double_tags.sub(lambda m: f'<{m.group(2)}>{m.group(4)}</{m.group(2)}>', processed_text)
    return processed_text

def remove_tail(processed_text):
    """
    Удаление в тексте хвоста после символов '(', '[',';', '<', '!'
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_tail.sub('', processed_text)
    processed_text = processed_text.strip()
    return processed_text


def remove_new_lines(processed_text, divider=''):
    """
    Удаление в начале тексте специальных символов
    :param processed_text: обрабатываемый текст
    :param divider: разделитель, который будет вставлен вместо переноса строки
    :return: обработанный текст
    """
    processed_text = _pattern_new_lines .sub(divider, processed_text)
    processed_text = processed_text.strip()
    return processed_text


def remove_name_prefix(processed_text):
    """
    Удаление в начале текста специальных символов
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_name_prefix1.sub('', processed_text)
    processed_text = _pattern_name_prefix2.sub(r'\1', processed_text)
    processed_text = _pattern_name_prefix3.sub('', processed_text)
    processed_text = processed_text.strip()
    return processed_text

def remove_name_postfix(processed_text):
    """
    Удаление в конце текста специальных символов
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_name_postfix.sub('', processed_text)
    processed_text = processed_text.strip()
    return processed_text
def remove_description_prefix(processed_text):
    """
    Удаление в начале тексте специальных символов
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_description_prefix.sub('', processed_text)
    processed_text = processed_text.strip()
    return processed_text

def remove_description_postfix(processed_text):
    """
    Удаление в начале тексте специальных символов
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_description_postfix.sub('', processed_text)
    processed_text = processed_text.strip()
    return processed_text

def remove_punctuation_marks(processed_text):
    """
    Удаление в тексте знаков пунктуации
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    chars = list(processed_text)
    punctuation_marks = ['!', ',', '(', ')', ':', '?', '..', '...', '«', '»', ';', '–', '--', '[', ']', '{', '}']
    new_chars = [char for char in chars if char not in punctuation_marks]
    processed_text = ' '.join(new_chars)
    return processed_text


def remove_between_square_brackets(processed_text):
    """
    Удаление текста в квадратных скобках
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    return _pattern_between_square_brackets.sub('', processed_text)


def remove_between_parentheses(processed_text):
    """
    Удаление текста в круглых скобках
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    return _pattern__between_parentheses.sub('', processed_text)


def replace_pstrong_tags(processed_text):
    """
    Замена сочетания тегов p и strong на h1
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = _pattern_pbr_open_tags.sub('<p>', processed_text)
    processed_text = _pattern_pbr_close_tags.sub('</p>', processed_text)
    processed_text = _pattern_pstrong_tags.sub(lambda m: f'<h1>{m.group(2)}</h1>', processed_text)
    processed_text = _pattern_double_tags.sub(lambda m: f'<{m.group(2)}>{m.group(4)}</{m.group(2)}>', processed_text)
    processed_text = _pattern_h_tags.sub('h1', processed_text)
    return processed_text


def divide_to_list(processed_text):
    processed_text = _pattern_list_dividers.sub('\n', processed_text)
    processed_text = processed_text.replace('\n\n\n','\n')
    processed_text = processed_text.replace('\n\n', '\n')
    return processed_text

def preprocess_name(processed_text):
    """
    Обработка текста названия объявления
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = processed_text.lower()
    processed_text = remove_between_square_brackets(processed_text)
    processed_text = remove_between_parentheses(processed_text)
    for word in stop_words_for_name:
        processed_text = processed_text.replace(word, '')
    processed_text = remove_name_prefix(processed_text)
    processed_text = processed_text.replace('-', ' ')
    processed_text = processed_text.replace('\\', '/')
    processed_text = processed_text.replace('\\\\', '/ ')
    processed_text = processed_text.replace('\u200b', ' ')
    processed_text = processed_text.replace('\u200e', ' ')
    processed_text = remove_tail(processed_text)
    processed_text = remove_emoji(processed_text)
    processed_text = remove_punctuation_marks(processed_text)
    processed_text = remove_whitespaces(processed_text)

    return processed_text


def preprocess_description(processed_text):
    """
    Обработка текста описания объявления
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    processed_text = processed_text.replace('\n', ' ')
    processed_text = processed_text.replace('\u200b', '')
    processed_text = processed_text.replace('\u200e', '')
    processed_text = remove_whitespaces(processed_text)
    processed_text = remove_description_postfix(processed_text)
    processed_text = remove_description_prefix(processed_text)
    processed_text = remove_tag_whitespaces(processed_text)

    ln_new = len(processed_text)
    ln_old = ln_new+1
    while ln_old > ln_new:
        processed_text = remove_empty_tags(processed_text)
        ln_old = ln_new
        ln_new = len(processed_text)

    processed_text = remove_whitespaces(processed_text)
    processed_text = remove_duplicate_tags(processed_text)
    processed_text = processed_text.replace('div', 'p')
    processed_text = replace_pstrong_tags(processed_text)
    processed_text = _pattern_between_par_tag_whitespaces.sub(lambda m: m.group(2), processed_text)


    return processed_text

def preprocess_listitem(processed_text):
    """
    Обработка заголовка и текста блока
    :param processed_text: обрабатываемый текст
    :return: обработанный текст
    """
    # processed_text = processed_text.lower()
    processed_text = remove_double_symbols(processed_text)
    processed_text = remove_name_postfix(processed_text)
    processed_text = remove_name_prefix(processed_text)
    processed_text = processed_text.replace('-', ' ')
    processed_text = processed_text.replace('\\', '/')
    processed_text = processed_text.replace('\\\\', '/ ')
    processed_text = processed_text.replace('\u200b', ' ')
    processed_text = processed_text.replace('\u200e', ' ')
    processed_text = remove_emoji(processed_text)
    processed_text = remove_url(processed_text)
    processed_text = remove_emails(processed_text)
    processed_text = remove_bots(processed_text)
    processed_text = remove_punkt_symbols(processed_text)
    processed_text = remove_nolist_characters(processed_text)

    processed_text = remove_whitespaces(processed_text)

    return processed_text
