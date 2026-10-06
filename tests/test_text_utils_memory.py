from unittest.mock import patch
from data_model import CharacterEntry, EntityEntry, SessionMemory
from text_utils import update_session_memory, get_formatted_user_prompt, get_formatted_user_prompt_with_image
import tempfile, os
from text_utils import get_formatted_user_prompt_plain, translate, fallback_translation


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



def _sample_memory() -> SessionMemory:
    return SessionMemory(
        characters=[CharacterEntry(original_name="田中", translated_name="Tanaka", gender="male", notes="protagonist")],
        places=[EntityEntry(original="新宿", translated="Shinjuku")],
        organizations=[EntityEntry(original="黒烏", translated="Black Crow Corp")],
        story_summary="Tanaka arrived in Shinjuku.",
    )


def test_prompt_includes_memory_section():
    prompt = get_formatted_user_prompt(
        context="page context",
        text="田中はどこだ？",
        source_language="Japanese",
        target_language="English",
        session_memory=_sample_memory(),
    )
    assert "Tanaka (male" in prompt
    assert "Shinjuku" in prompt
    assert "Black Crow Corp" in prompt
    assert "Tanaka arrived in Shinjuku." in prompt


def test_prompt_excludes_memory_section_when_empty():
    prompt = get_formatted_user_prompt(
        context="page context",
        text="田中はどこだ？",
        source_language="Japanese",
        target_language="English",
        session_memory=SessionMemory(),
    )
    assert "Known entities" not in prompt
    assert "Story so far" not in prompt


def test_prompt_excludes_memory_section_when_none():
    prompt = get_formatted_user_prompt(
        context="page context",
        text="田中はどこだ？",
        source_language="Japanese",
        target_language="English",
    )
    assert "Known entities" not in prompt


def test_prompt_with_image_includes_memory_section():
    prompt = get_formatted_user_prompt_with_image(
        context="page context",
        text="田中はどこだ？",
        source_language="Japanese",
        target_language="English",
        session_memory=_sample_memory(),
    )
    assert "Tanaka (male" in prompt
    assert "Tanaka arrived in Shinjuku." in prompt


def test_memory_injected_before_previous_translations():
    prev = [{"original": "こんにちは", "translated": "Hello"}]
    prompt = get_formatted_user_prompt(
        context="ctx",
        text="田中はどこだ？",
        source_language="Japanese",
        target_language="English",
        previous_translations=prev,
        session_memory=_sample_memory(),
    )
    memory_pos = prompt.index("Known entities")
    prev_pos = prompt.index("Previous translations")
    assert memory_pos < prev_pos


def test_memory_injected_before_previous_translations_with_image():
    prev = [{"original": "こんにちは", "translated": "Hello"}]
    prompt = get_formatted_user_prompt_with_image(
        context="ctx",
        text="田中はどこだ？",
        source_language="Japanese",
        target_language="English",
        previous_translations=prev,
        session_memory=_sample_memory(),
    )
    memory_pos = prompt.index("Known entities")
    prev_pos = prompt.index("Previous translations")
    assert memory_pos < prev_pos


def test_prompt_includes_summary_when_only_summary_set():
    mem = SessionMemory(story_summary="Tanaka arrived in Shinjuku.")
    prompt = get_formatted_user_prompt(
        context="ctx",
        text="田中はどこだ？",
        source_language="Japanese",
        target_language="English",
        session_memory=mem,
    )
    assert "Story so far: Tanaka arrived in Shinjuku." in prompt
    assert "Known entities" not in prompt


def test_plain_prompt_has_output_only_instruction():
    prompt = get_formatted_user_prompt_plain(
        context="ctx",
        text="こんにちは",
        source_language="Japanese",
        target_language="English",
    )
    assert "Output ONLY the translated text" in prompt
    assert "JSON" not in prompt


def test_translate_plain_calls_llm_without_format():
    with patch("text_utils.call_llm", return_value="Hello") as mock_llm:
        result = translate(
            "こんにちは",
            model="test-model",
            context="ctx",
            source_language="Japanese",
            use_json=False,
        )
    assert mock_llm.call_args.kwargs.get("format") is None
    assert result == "Hello"


def test_translate_plain_no_fallback():
    with patch("text_utils.call_llm", return_value="this is a translation"), \
         patch("text_utils.fallback_translation") as mock_fallback:
        translate(
            "こんにちは",
            model="test-model",
            context="ctx",
            source_language="Japanese",
            use_json=False,
        )
    mock_fallback.assert_not_called()


def test_update_session_memory_keeps_entries_the_model_drops():
    existing = _sample_memory()
    reply = SessionMemory(story_summary="Tanaka looked for someone.").model_dump_json()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("text_utils.call_llm", return_value=reply):
            result = update_session_memory(_make_translations(), existing, "test-model", tmpdir)
    assert result.characters == existing.characters
    assert result.organizations == existing.organizations
    assert result.story_summary == "Tanaka looked for someone."


def test_update_session_memory_rejects_entities_not_on_page():
    reply = SessionMemory(
        characters=[CharacterEntry(original_name="鈴木", translated_name="Suzuki", gender="male")],
    ).model_dump_json()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("text_utils.call_llm", return_value=reply):
            result = update_session_memory(_make_translations(), SessionMemory(), "test-model", tmpdir)
    assert result.characters == []


def test_update_session_memory_prompt_lists_known_names():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("text_utils.call_llm", return_value="{}") as mock_llm:
            update_session_memory(_make_translations(), _sample_memory(), "test-model", tmpdir)
    user_prompt = mock_llm.call_args.args[2]
    assert "Known names (do not repeat): 田中, 新宿, 黒烏" in user_prompt
    assert mock_llm.call_args.kwargs["num_ctx"] >= 8192


def test_translate_sends_only_entities_in_context():
    reply = '{"input_text": "x", "translated_text": "Where is Tanaka?"}'
    with patch("text_utils.call_llm", return_value=reply) as mock_llm:
        translate("田中はどこだ？", model="m", context="[Current Page] 田中はどこだ？", source_language="Japanese",
                  session_memory=_sample_memory())
    user_prompt = mock_llm.call_args.args[2]
    assert "田中 → Tanaka" in user_prompt
    assert "新宿 →" not in user_prompt
    assert "黒烏 →" not in user_prompt
    assert "Story so far: Tanaka arrived in Shinjuku." in user_prompt


def test_translate_passes_relevant_memory_to_fallback():
    with patch("text_utils.call_llm", return_value="</translation>"), \
         patch("text_utils.fallback_translation", return_value="Where is Tanaka?") as mock_fallback:
        translate("田中はどこだ？", model="m", context="ctx", source_language="Japanese",
                  session_memory=_sample_memory(), use_json=False)
    memory = mock_fallback.call_args.kwargs["session_memory"]
    assert [c.translated_name for c in memory.characters] == ["Tanaka"]
    assert memory.places == []


def test_fallback_prompt_includes_glossary():
    with patch("text_utils.call_llm", return_value="Where is Tanaka?</translation>") as mock_llm:
        fallback_translation("田中はどこだ？", "m", "Japanese", "English", session_memory=_sample_memory())
    user_prompt = mock_llm.call_args.args[2]
    assert "- 田中 → Tanaka (male, protagonist)" in user_prompt
    assert "Story so far" not in user_prompt
    assert user_prompt.rstrip().endswith("</translation>")


def test_fallback_prompt_without_memory_has_no_glossary():
    with patch("text_utils.call_llm", return_value="Hello</translation>") as mock_llm:
        fallback_translation("こんにちは", "m", "Japanese", "English")
    assert "names" not in mock_llm.call_args.args[2]


def test_update_session_memory_disables_repetition_penalties():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("text_utils.call_llm", return_value="{}") as mock_llm:
            update_session_memory(_make_translations(), SessionMemory(), "test-model", tmpdir)
    assert mock_llm.call_args.kwargs["presence_penalty"] == 0.0
    assert mock_llm.call_args.kwargs["frequency_penalty"] == 0.0
