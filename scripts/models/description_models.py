from pydantic import BaseModel

from scripts.models.enums import ContentTypes, SemanticTypes


class BlockData(BaseModel):
    """
    Структура для сохранения результатов парсинга блоков
    """
    block_id: int = 0
    id_hh: int = 0
    title: str = None
    content: str = None
    content_type: int = ContentTypes.text.value
    semantic_type: int = SemanticTypes.unknown.value


class ParseResult(BaseModel):
    """
    Структура для сохранения результатов обработки вакансии
    """
    is_success: bool = False
    message: str = None
    key_skills: set[str] = None
    add_skills: set[str] = None
    content: list[BlockData] = None
