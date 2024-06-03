import unittest
import json

from scripts.model_scripts.process_vacancy import process_vacancy
from tests.params_t import serialized_text, html_text

# Тестовые данные
text2 = {"text": serialized_text}
txt3 = {"text": html_text}
txt4 = json.dumps(txt3)

if __name__ == '__main__':
    unittest.main()


def test_process_vacancy_true_result():
    """Тестирование обработки вакансии при корректных исходных данных.
    Проверка флага результата и сообщения об ошибке"""
    result = process_vacancy(html_text)
    assert result.is_success == True
    assert result.message is None


def test_process_vacancy_get_key_skill():
    """Тестирование обработки вакансии при корректных исходных данных."""
    result = process_vacancy(html_text)
    assert len(result.key_skills) > 0
    assert 'kotlin' in result.key_skills


def test_process_vacancy_get_add_skills():
    """Тестирование обработки вакансии при корректных исходных данных."""
    result = process_vacancy(html_text)
    assert len(result.add_skills) > 0
    assert 'redis' in result.add_skills


def test_process_vacancy_get_content():
    """Тестирование обработки вакансии при корректных исходных данных."""
    result = process_vacancy(html_text)
    assert len(result.content) == 6
    assert result.content[0].block_id == 1 and result.content[0].semantic_type == 0
    assert result.content[2].block_id == 4 and result.content[2].semantic_type == 1
    assert result.content[3].block_id == 6 and result.content[3].semantic_type == 1


def test_process_vacancy_empty_text_error():
    """
    Тестирование обработки вакансии при некорректных исходных данных.
    Передача пустой строки
    """
    result = process_vacancy("1")
    assert result.is_success == False
    assert result.message == "Передан пустой текст"


def test_process_vacancy_not_text_error():
    """
    Тестирование обработки вакансии при некорректных исходных данных.
    Передача данных неверного рипа
    """
    result = process_vacancy(1)
    assert result.is_success == False
    assert result.message.startswith("Неверный тип данных")


def test_process_vacancy_not_html_text_error():
    """
    Тестирование обработки вакансии при некорректных исходных данных.
    Переданный текст не отфарматирован с помощью html
    """
    result = process_vacancy("true html")
    assert result.is_success is False
    assert result.message == "Поле описания должно иметь html разметку"
