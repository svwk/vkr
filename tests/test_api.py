import sys
import json
from fastapi.testclient import TestClient

from tests.params_t import html_text
from api import app

sys.path.append('.')
client = TestClient(app)

txt3 = {"text": html_text}
txt4 = json.dumps(txt3)


def test_api_get():
    """Тестирование доступа к сервису"""
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {
        "message": "Извлечение элементов структуры из текста вакансии, оформленной с помощью HTML-разметки"}


def test_api_process_vacancy():
    """
    Тестирование енд-пойнта для запуска обработки вакансии при корректных исходных данных.
    Должен вернуть код 200
    """
    req = client.post("/process-vacancy/", json=txt3)
    assert req.status_code == 200
    assert req.json() != ""
    json_data = req.json()
    assert json_data['is_success'] is True


def test_api_process_vacancy_empty_text_error():
    """
    Тестирование енд-пойнта для запуска обработки вакансии при некорректных исходных данных.
    Должен вернуть код 400
    """
    req = client.post("/process-vacancy/", json={"text": "1"})
    assert req.status_code == 400
    json_data = req.json()

    assert json_data['is_success'] is False
    assert json_data['message'] == "Передан пустой текст"
