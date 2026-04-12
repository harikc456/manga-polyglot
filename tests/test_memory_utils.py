import pytest
from pydantic import ValidationError
from data_model import CharacterEntry, EntityEntry, SessionMemory


def test_character_entry_defaults():
    entry = CharacterEntry(original_name="田中", translated_name="Tanaka", gender="male")
    assert entry.notes == ""


def test_entity_entry():
    entry = EntityEntry(original="新宿", translated="Shinjuku")
    assert entry.original == "新宿"
    assert entry.translated == "Shinjuku"


def test_session_memory_defaults():
    mem = SessionMemory()
    assert mem.characters == []
    assert mem.places == []
    assert mem.organizations == []
    assert mem.story_summary == ""


def test_session_memory_json_schema_has_required_fields():
    schema = SessionMemory.model_json_schema()
    props = schema.get("properties", {})
    assert "characters" in props
    assert "places" in props
    assert "organizations" in props
    assert "story_summary" in props


def test_character_entry_requires_name_and_gender():
    with pytest.raises(ValidationError):
        CharacterEntry(original_name="田中")  # missing translated_name and gender


def test_entity_entry_requires_both_fields():
    with pytest.raises(ValidationError):
        EntityEntry(original="新宿")  # missing translated
