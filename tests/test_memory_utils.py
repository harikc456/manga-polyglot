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


import os
import tempfile
from memory_utils import serialize_memory, load_memory, format_memory_for_prompt


def _sample_memory() -> SessionMemory:
    return SessionMemory(
        characters=[
            CharacterEntry(original_name="田中", translated_name="Tanaka", gender="male", notes="protagonist"),
            CharacterEntry(original_name="桜", translated_name="Sakura", gender="female"),
        ],
        places=[EntityEntry(original="新宿", translated="Shinjuku")],
        organizations=[EntityEntry(original="黒烏", translated="Black Crow Corp")],
        story_summary="Tanaka discovers a hidden door.",
    )


def test_serialize_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "memory.md")
        mem = _sample_memory()
        serialize_memory(mem, path)
        loaded = load_memory(path)

    assert loaded.story_summary == "Tanaka discovers a hidden door."
    assert len(loaded.characters) == 2
    assert loaded.characters[0].original_name == "田中"
    assert loaded.characters[0].translated_name == "Tanaka"
    assert loaded.characters[0].gender == "male"
    assert loaded.characters[0].notes == "protagonist"
    assert loaded.characters[1].gender == "female"
    assert len(loaded.places) == 1
    assert loaded.places[0].original == "新宿"
    assert loaded.places[0].translated == "Shinjuku"
    assert len(loaded.organizations) == 1
    assert loaded.organizations[0].translated == "Black Crow Corp"


def test_load_memory_missing_file():
    loaded = load_memory("/tmp/does_not_exist_xyz.md")
    assert loaded == SessionMemory()


def test_format_memory_for_prompt_empty():
    result = format_memory_for_prompt(SessionMemory())
    assert result == ""


def test_format_memory_for_prompt_populated():
    mem = _sample_memory()
    result = format_memory_for_prompt(mem)
    assert "Tanaka (male" in result
    assert "Sakura (female" in result
    assert "Shinjuku" in result
    assert "Black Crow Corp" in result
    assert "Tanaka discovers a hidden door." in result


def test_serialize_creates_readable_markdown():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "memory.md")
        serialize_memory(_sample_memory(), path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    assert "## Story Summary" in content
    assert "## Characters" in content
    assert "## Places" in content
    assert "## Organizations" in content
    assert "田中" in content
    assert "Tanaka" in content
