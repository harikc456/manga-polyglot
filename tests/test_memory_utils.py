import os
import tempfile

import pytest
from pydantic import ValidationError
from data_model import CharacterEntry, EntityEntry, SessionMemory
from memory_utils import serialize_memory, load_memory, format_memory_for_prompt, merge_memory, filter_memory, format_glossary


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


def test_format_memory_for_prompt_pairs_originals_with_translations():
    result = format_memory_for_prompt(_sample_memory())
    assert "田中 → Tanaka (male" in result
    assert "新宿 → Shinjuku" in result


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


def test_serialize_load_roundtrip_with_pipe_in_notes():
    mem = SessionMemory(
        characters=[CharacterEntry(original_name="田中", translated_name="Tanaka", gender="male", notes="hero | protagonist")],
        places=[],
        organizations=[],
        story_summary="",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "memory.md")
        serialize_memory(mem, path)
        loaded = load_memory(path)
    assert loaded.characters[0].notes == "hero | protagonist"


def test_format_memory_for_prompt_one_entry_per_line():
    lines = format_memory_for_prompt(_sample_memory()).splitlines()
    assert "- 田中 → Tanaka (male, protagonist)" in lines
    assert "- 桜 → Sakura (female)" in lines
    assert "- 新宿 → Shinjuku" in lines


def _char(original, translated, gender="unknown", notes=""):
    return CharacterEntry(original_name=original, translated_name=translated, gender=gender, notes=notes)


def test_merge_keeps_entries_missing_from_update():
    merged = merge_memory(_sample_memory(), SessionMemory(story_summary="New summary."), "なにもない")
    assert [c.translated_name for c in merged.characters] == ["Tanaka", "Sakura"]
    assert merged.places == _sample_memory().places
    assert merged.organizations == _sample_memory().organizations
    assert merged.story_summary == "New summary."


def test_merge_never_renames_known_entities():
    update = SessionMemory(
        characters=[_char("田中", "Mr. Tanaka", "female")],
        places=[EntityEntry(original="新宿", translated="Shinjuku Ward")],
    )
    merged = merge_memory(_sample_memory(), update, "田中は新宿にいる")
    assert merged.characters[0].translated_name == "Tanaka"
    assert merged.characters[0].gender == "male"
    assert merged.places == [EntityEntry(original="新宿", translated="Shinjuku")]


def test_merge_fills_unknown_gender_and_empty_notes():
    memory = SessionMemory(characters=[_char("桜", "Sakura")])
    merged = merge_memory(memory, SessionMemory(characters=[_char("桜", "Sakura-chan", "female", "classmate")]), "")
    assert merged.characters[0] == _char("桜", "Sakura", "female", "classmate")
    assert memory.characters[0].gender == "unknown"  # input memory is not mutated


def test_merge_adds_new_entities_found_on_page():
    update = SessionMemory(
        characters=[_char("佐藤", "Sato", "female")],
        places=[EntityEntry(original="渋谷", translated="Shibuya")],
        organizations=[EntityEntry(original="白狐", translated="White Fox")],
    )
    merged = merge_memory(SessionMemory(), update, "佐藤は渋谷で白狐に会った")
    assert merged.characters == [_char("佐藤", "Sato", "female")]
    assert merged.places == [EntityEntry(original="渋谷", translated="Shibuya")]
    assert merged.organizations == [EntityEntry(original="白狐", translated="White Fox")]


def test_merge_rejects_entities_not_on_page():
    update = SessionMemory(
        characters=[_char("鈴木", "Suzuki", "male")],
        places=[EntityEntry(original="大阪", translated="Osaka")],
    )
    merged = merge_memory(SessionMemory(), update, "佐藤は渋谷にいる")
    assert merged.characters == []
    assert merged.places == []


@pytest.mark.parametrize("original,translated", [("", "Nobody"), ("佐藤", " "), ("佐藤" * 11, "Sato" * 11)])
def test_merge_rejects_empty_or_overlong_entities(original, translated):
    update = SessionMemory(characters=[_char(original, translated)])
    assert merge_memory(SessionMemory(), update, "佐藤" * 20).characters == []


def test_merge_matches_names_across_width_and_whitespace():
    update = SessionMemory(characters=[_char("ＡＢＣ", "ABC"), _char("ABC", "Abc")])
    merged = merge_memory(SessionMemory(), update, "A B C が来た")
    assert merged.characters == [_char("ＡＢＣ", "ABC")]  # accepted once, duplicate dropped


def test_merge_keeps_summary_when_update_has_none():
    assert merge_memory(_sample_memory(), SessionMemory(), "").story_summary == "Tanaka discovers a hidden door."


def test_filter_memory_keeps_only_entities_in_text():
    filtered = filter_memory(_sample_memory(), "[Current Page] 桜、新宿に行こう")
    assert [c.original_name for c in filtered.characters] == ["桜"]
    assert [p.original for p in filtered.places] == ["新宿"]
    assert filtered.organizations == []
    assert filtered.story_summary == "Tanaka discovers a hidden door."


def test_format_glossary_lists_all_entities():
    assert format_glossary(_sample_memory()) == [
        "田中 → Tanaka (male, protagonist)",
        "桜 → Sakura (female)",
        "新宿 → Shinjuku",
        "黒烏 → Black Crow Corp",
    ]


def test_merge_treats_honorific_variant_as_known_character():
    memory = SessionMemory(characters=[_char("田中", "Tanaka")])
    update = SessionMemory(characters=[_char("田中先輩", "Tanaka-senpai", "male")])
    merged = merge_memory(memory, update, "田中先輩！待って")
    assert merged.characters == [_char("田中", "Tanaka", "male")]


def test_merge_keeps_name_that_is_only_an_honorific_word():
    update = SessionMemory(characters=[_char("先生", "Teacher")])
    assert merge_memory(SessionMemory(), update, "先生が来た").characters == [_char("先生", "Teacher")]


def test_merge_drops_rambling_notes():
    update = SessionMemory(characters=[_char("桜", "Sakura", "female", "x" * 200), _char("佐藤", "Sato", notes="rival")])
    merged = merge_memory(SessionMemory(), update, "桜と佐藤")
    assert [c.notes for c in merged.characters] == ["", "rival"]


def test_filter_memory_matches_character_with_honorific():
    memory = SessionMemory(characters=[_char("田中先輩", "Tanaka")])
    assert filter_memory(memory, "田中さん、おはよう").characters == memory.characters
