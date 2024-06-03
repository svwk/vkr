from fastapi import APIRouter, File,  Body, Response, status
from typing import Annotated

from scripts.model_scripts import process_vacancy
from scripts.models.description_models import ParseResult


api_router = APIRouter()


@api_router.get("/")
async def root():
    return {"message": "Извлечение элементов структуры из текста вакансии, оформленной с помощью HTML-разметки"}


@api_router.post("/process-vacancy-file/", status_code=200)
def process_file(vacancy_file: bytes = File(), response: Response = status.HTTP_200_OK) -> ParseResult:
    """ Выделение элементов структуры из тела вакансии с html-разметкой,
    а также извлечение требований к соискателю
    - **vacancy_file**: Файл с текстом вакансии в html-разметке
    """
    try:
        text = vacancy_file.decode('utf-8')
        result = process_vacancy.process_vacancy(text)
        if not result.is_success:
            response.status_code = status.HTTP_400_BAD_REQUEST
        return result

    except BaseException as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ParseResult(message=str(e))


@api_router.post("/process-vacancy/", status_code=200)
def process(text: Annotated[str, Body(embed=True)], response: Response = status.HTTP_200_OK) -> ParseResult:
    """ Выделение элементов структуры из тела вакансии с html-разметкой,
    а также извлечение требований к соискателю
    - **text**: Текст вакансии в html-разметке
    """
    try:
        result = process_vacancy.process_vacancy(text)
        if not result.is_success:
            response.status_code = status.HTTP_400_BAD_REQUEST
        return result

    except BaseException as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ParseResult(message=str(e))
