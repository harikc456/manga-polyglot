from pydantic import BaseModel


class Translation(BaseModel):
    input_text: str
    translated_texts: list[str]
    notes: str | None
