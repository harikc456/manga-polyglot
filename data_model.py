from typing import Literal
from pydantic import BaseModel, field_validator


class Translation(BaseModel):
    input_text: str
    translated_text: str


class EntityEntry(BaseModel):
    original: str
    translated: str


class CharacterEntry(BaseModel):
    original_name: str
    translated_name: str
    gender: Literal["male", "female", "unknown"]
    notes: str = ""

    @field_validator("gender", mode="before")
    @classmethod
    def coerce_gender(cls, v: object) -> str:
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ("male", "m", "man", "boy"):
                return "male"
            if v_lower in ("female", "f", "woman", "girl", "woman"):
                return "female"
        return "unknown"


class SessionMemory(BaseModel):
    characters: list[CharacterEntry] = []
    places: list[EntityEntry] = []
    organizations: list[EntityEntry] = []
    story_summary: str = ""
