from unittest.mock import patch

import pytest

from text_utils import TRANSLATE_NUM_CTX, clean_ocr_garbage, get_formatted_user_prompt_plain, translate


@pytest.mark.parametrize("text", ["コーヒー", "サッカー", "ねー", "ーあ"])
def test_clean_ocr_garbage_keeps_long_vowel_mark(text):
    assert clean_ocr_garbage(text) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("すごーい！！！", "すごーい！！！"),
        ("えっ……", "えっ……"),
        ("なに？！？", "なに？！？"),
        ("What!!!", "What!!!"),
        ("……", "……"),
    ],
)
def test_clean_ocr_garbage_keeps_expressive_punctuation(text, expected):
    assert clean_ocr_garbage(text) == expected


def test_clean_ocr_garbage_caps_long_ellipsis_runs():
    assert clean_ocr_garbage("えっ……………") == "えっ………"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("こんにちは┌┐└┘", "こんにちは"),
        ("こんにちは@#$%", "こんにちは"),
        ("​こんにちは﻿", "こんにちは"),
    ],
)
def test_clean_ocr_garbage_still_removes_junk(text, expected):
    assert clean_ocr_garbage(text) == expected


def _translate(**kwargs):
    return translate("こんにちは", model="m", context="ctx", source_language="Japanese", **kwargs)


def test_translate_json_falls_back_on_invalid_json():
    with patch("text_utils.call_llm", return_value="not json"), \
         patch("text_utils.fallback_translation", return_value="Hello") as mock_fallback:
        assert _translate(use_json=True) == "Hello"
    mock_fallback.assert_called_once()


def test_translate_json_falls_back_on_schema_mismatch():
    with patch("text_utils.call_llm", return_value='{"translated_text": 5}'), \
         patch("text_utils.fallback_translation", return_value="Hello") as mock_fallback:
        assert _translate(use_json=True) == "Hello"
    mock_fallback.assert_called_once()


@pytest.mark.parametrize("use_json", [True, False])
def test_translate_uses_large_context_window(use_json):
    response = '{"input_text": "こんにちは", "translated_text": "Hello"}' if use_json else "Hello"
    with patch("text_utils.call_llm", return_value=response) as mock_llm:
        _translate(use_json=use_json)
    assert mock_llm.call_args.kwargs["num_ctx"] == TRANSLATE_NUM_CTX
    assert TRANSLATE_NUM_CTX >= 8192


# --- source languages other than Japanese -----------------------------------

def test_translate_sends_korean_source_to_llm():
    with patch("text_utils.call_llm", return_value="Hello") as mock_llm:
        result = translate("안녕하세요", model="m", context="ctx", source_language="Korean", use_json=False)
    mock_llm.assert_called_once()
    assert result == "Hello"


def test_translate_keeps_spaces_for_non_japanese_source():
    with patch("text_utils.call_llm", return_value="Hello friend") as mock_llm:
        translate("안녕 친구", model="m", context="ctx", source_language="Korean", use_json=False)
    assert "<text>안녕 친구</text>" in mock_llm.call_args.args[2]


def test_translate_skips_llm_for_text_without_letters():
    with patch("text_utils.call_llm") as mock_llm:
        assert translate("！？", model="m", context="ctx", source_language="Korean") == "!?"
    mock_llm.assert_not_called()


def test_translate_japanese_source_still_skips_untranslatable_text():
    with patch("text_utils.call_llm") as mock_llm:
        assert translate("OK", model="m", context="ctx", source_language="Japanese") == "OK"
    mock_llm.assert_not_called()


# --- fallback consistency ----------------------------------------------------

@pytest.mark.parametrize("use_json", [True, False])
def test_translate_falls_back_when_output_is_still_japanese(use_json):
    response = '{"input_text": "x", "translated_text": "こんにちは"}' if use_json else "こんにちは"
    with patch("text_utils.call_llm", return_value=response), \
         patch("text_utils.fallback_translation", return_value="Hello") as mock_fallback:
        assert _translate(use_json=use_json) == "Hello"
    mock_fallback.assert_called_once()


@pytest.mark.parametrize("use_json", [True, False])
def test_translate_falls_back_on_empty_output(use_json):
    response = '{"input_text": "x", "translated_text": ""}' if use_json else ""
    with patch("text_utils.call_llm", return_value=response), \
         patch("text_utils.fallback_translation", return_value="Hello") as mock_fallback:
        assert _translate(use_json=use_json) == "Hello"
    mock_fallback.assert_called_once()


# --- tagged output (plain mode and fallback) -----------------------------------

def test_plain_prompt_asks_for_translation_tags():
    prompt = get_formatted_user_prompt_plain("ctx", "こんにちは", "Japanese", "English")
    assert "<translation>" in prompt and "</translation>" in prompt


def test_translate_plain_prefills_tag_and_stops_at_closing_tag():
    with patch("text_utils.call_llm", return_value="Hello") as mock_llm:
        assert _translate(use_json=False) == "Hello"
    assert mock_llm.call_args.kwargs["prefill"] == "<translation>"
    assert "</translation>" in mock_llm.call_args.kwargs["stop"]


@pytest.mark.parametrize(
    "response",
    [
        "Hello",                                         # prefill + stop honoured
        "Hello</translation>\nNote: casual register.",   # stop sequence not honoured
        "<translation>Hello</translation>",              # prefill echoed back
        "<translation>Hello",
    ],
)
def test_translate_plain_keeps_only_tagged_text(response):
    with patch("text_utils.call_llm", return_value=response), \
         patch("text_utils.fallback_translation") as mock_fallback:
        assert _translate(use_json=False) == "Hello"
    mock_fallback.assert_not_called()


def test_translate_plain_falls_back_on_empty_tag():
    with patch("text_utils.call_llm", return_value="</translation>"), \
         patch("text_utils.fallback_translation", return_value="Hello") as mock_fallback:
        assert _translate(use_json=False) == "Hello"
    mock_fallback.assert_called_once()


def test_fallback_uses_tagged_output():
    from text_utils import fallback_translation

    with patch("text_utils.call_llm", return_value="Hello</translation> (literal)") as mock_llm:
        assert fallback_translation("こんにちは", "m", "Japanese", "English") == "Hello"
    assert mock_llm.call_args.kwargs["prefill"] == "<translation>"
    assert "</translation>" in mock_llm.call_args.kwargs["stop"]
    assert "<translation>" in mock_llm.call_args.args[2]


def test_call_llm_sends_prefill_as_assistant_message():
    from text_utils import call_llm

    with patch("text_utils.chat") as mock_chat:
        mock_chat.return_value.message.content = "Hello"
        call_llm("m", "sys", "user", prefill="<translation>")
    messages = mock_chat.call_args.kwargs["messages"]
    assert messages[-1] == {"role": "assistant", "content": "<translation>"}


def test_call_llm_without_prefill_ends_with_user_message():
    from text_utils import call_llm

    with patch("text_utils.chat") as mock_chat:
        mock_chat.return_value.message.content = "Hello"
        call_llm("m", "sys", "user")
    assert mock_chat.call_args.kwargs["messages"][-1]["role"] == "user"


def test_translate_strips_translation_prefix_without_fallback():
    with patch("text_utils.call_llm", return_value="Translation: Hello"), \
         patch("text_utils.fallback_translation") as mock_fallback:
        assert _translate(use_json=False) == "Hello"
    mock_fallback.assert_not_called()


@pytest.mark.parametrize("output", ["Translate this, now!", "Lost in translation...", "This is a translation"])
def test_translate_accepts_dialogue_mentioning_translation(output):
    response = f'{{"input_text": "x", "translated_text": "{output}"}}'
    with patch("text_utils.call_llm", return_value=response), \
         patch("text_utils.fallback_translation") as mock_fallback:
        assert _translate(use_json=True) == output
    mock_fallback.assert_not_called()


def test_fallback_prompt_names_source_language():
    from text_utils import fallback_translation

    with patch("text_utils.call_llm", return_value="Hello") as mock_llm:
        fallback_translation("こんにちは", "m", "Japanese", "English")
    assert "text in Japanese to English" in mock_llm.call_args.args[1]


# --- JSON prompt matches the schema -------------------------------------------

def test_json_prompts_name_schema_fields():
    from data_model import Translation
    from text_utils import get_formatted_user_prompt, get_formatted_user_prompt_with_image

    for build in (get_formatted_user_prompt, get_formatted_user_prompt_with_image):
        prompt = build("ctx", "こんにちは", "Japanese", "English")
        for field in Translation.model_fields:
            assert f"- {field} -" in prompt
        assert "- text -" not in prompt
