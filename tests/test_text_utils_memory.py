from unittest.mock import patch
from data_model import CharacterEntry, EntityEntry, SessionMemory
from text_utils import update_session_memory
import tempfile, os


def _make_translations():
    return [
        {"original": "田中はどこだ？", "translated": "Where is Tanaka?"},
        {"original": "新宿に行った。", "translated": "He went to Shinjuku."},
    ]


def test_update_session_memory_returns_session_memory():
    updated = SessionMemory(
        characters=[CharacterEntry(original_name="田中", translated_name="Tanaka", gender="male")],
        places=[EntityEntry(original="新宿", translated="Shinjuku")],
        organizations=[],
        story_summary="Tanaka went to Shinjuku.",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("text_utils.call_llm", return_value=updated.model_dump_json()):
            result = update_session_memory(_make_translations(), SessionMemory(), "test-model", tmpdir)

    assert isinstance(result, SessionMemory)
    assert result.characters[0].translated_name == "Tanaka"
    assert result.story_summary == "Tanaka went to Shinjuku."


def test_update_session_memory_writes_memory_md():
    updated = SessionMemory(
        characters=[CharacterEntry(original_name="田中", translated_name="Tanaka", gender="male")],
        places=[],
        organizations=[],
        story_summary="Tanaka went to Shinjuku.",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("text_utils.call_llm", return_value=updated.model_dump_json()):
            update_session_memory(_make_translations(), SessionMemory(), "test-model", tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "memory.md"))


def test_update_session_memory_falls_back_on_invalid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("text_utils.call_llm", return_value="NOT VALID JSON"):
            original = SessionMemory(story_summary="original summary")
            result = update_session_memory(_make_translations(), original, "test-model", tmpdir)
    assert result.story_summary == "original summary"


def test_update_session_memory_falls_back_on_schema_violating_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("text_utils.call_llm", return_value='{"characters": null}'):
            original = SessionMemory(story_summary="original summary")
            result = update_session_memory(_make_translations(), original, "test-model", tmpdir)
    assert result.story_summary == "original summary"
