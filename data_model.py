from enum import Enum
from pydantic import BaseModel


class Translation(BaseModel):
    input_text: str
    translated_text: str

class BubbleType(str, Enum):
    FREE = "free"
    FIXED = "fixed"


class EntityEntry(BaseModel):
    original: str
    translated: str


class CharacterEntry(BaseModel):
    original_name: str
    translated_name: str
    gender: str   # "male" | "female" | "unknown"
    notes: str = ""


class SessionMemory(BaseModel):
    characters: list[CharacterEntry] = []
    places: list[EntityEntry] = []
    organizations: list[EntityEntry] = []
    story_summary: str = ""