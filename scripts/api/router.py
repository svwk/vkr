from fastapi import APIRouter, File

# from api_scripts.models import ClientData, BankDecisions
# from scripts.model_scripts.predict import predict as model_predict
from scripts.model_scripts import process_vacancy

api_router = APIRouter()


@api_router.get("/")
async def root():
    return {"message": "Извлечение элементов структуры  из текста вакансии, оформленной с помощью HTML-разметки"}


@api_router.post("/process-vacancy/")
def process(vacancy_file: bytes = File()):
    """ Выделение элементов структуры из тела вакансии с html-разметкой,
    а также извлечение требований к соискателю
    - **vacancy_file**: Файл с текстом вакансии в html-разметке
    """
    try:
        text = vacancy_file.decode('utf-8')
        return process_vacancy.process_vacancy(text)
    except BaseException as e:
        return {"Ошибка": f"{e}"}



