from enum import Enum


class ContentTypes(Enum):
    """
    Тип содержимого блока
    """
    # текстовый
    text: int = 0
    # список
    list: int = 1
    # возможно список
    list_br: int = 2
    #  только заголовок
    block_title: int = 3


class SemanticTypes(Enum):
    """
    Семантический тип
    """
    # Неизвестное назначение
    unknown: int = 0
    # описание вакансии
    description: int = 1
    # обязанности
    responsibilities: int = 2
    # требования
    requirements: int = 3
    # желательные требования
    desirable: int = 4
    # условия работы
    conditions: int = 5
