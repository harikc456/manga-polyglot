import re
import jaconv
import unicodedata
from ollama import chat, ChatResponse
from data_model import Translation

# Fixed System Prompt
SYSTEM_PROMPT = """
You are an experienced manga translator for one of the biggest publishers in the world. 
You will be given the entire context of a page or chapter and then will be asked to translate a specific text which is part of that context. 
You must use the context to provide an accurate translation of the text. 
Do not translate the context in one go. Ensure that the translated text remains meaningful taking the overall context into account.
Ensure the tenses and pronoun are consistent across the translation.

CRITICAL OUTPUT RULES:
- Output ONLY the translated text, nothing else
- DO NOT include explanations, breakdown, notes, or commentary about the translation
- The onomatopoeia needs to be translated according to how it sounds
- Similarly translate, voices for suprises like for example え？ being "Eh?" or "Huh"?
- The translated text is going to replace the original text, hence the length of the translation SHOULD be close the number of characters inside <text> tags
""".strip()


def is_japanese_char(char: str) -> bool:
    """Check if character is Hiragana, Katakana, or common Kanji."""
    code = ord(char)
    return (
        0x3040 <= code <= 0x309F  # Hiragana
        or 0x30A0 <= code <= 0x30FF  # Katakana
        or 0x4E00 <= code <= 0x9FFF  # Common Kanji
    )


def contains_japanese(text: str) -> bool:
    """Return True if any character in text is Japanese."""
    return any(is_japanese_char(char) for char in text)


def get_formatted_user_prompt(
    context: str, text: str, source_language: str, target_language: str
) -> str:
    return f"""Translate this {source_language} text from the manga to {target_language}.

Context: <context>{context}</context>

Text to translate: <text>{text}</text>

- You are to return JSON structure output with two fields
    - text - the original text that was supposed to be translated which would be {text}
    - translated_texts - a list of possible translations for the input text in {target_language}.
    - notes - any notes about the translation in less than 20 words
""".strip()


def clean_translated_text(text: str) -> str:
    """Clean model output: remove newlines, XML tags, quotes, and prefixes."""

    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"<.*?>", "", text)  # Remove XML/HTML tags
    text = re.sub(r'^"|"$', "", text)  # Strip surrounding quotes
    if text.lower().startswith("translation:"):
        text = text[len("translation:") :].lstrip(" :").strip()
    return text.strip()


def post_process(text: str) -> str:
    """Post-process Japanese text: normalize spaprint(translation.translated_texts)cing, ellipses, and half-width chars."""
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub(r"[・.]{2,}", lambda m: "." * len(m.group()), text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    num_ctx: int = 256,
    frequency_penalty: float = 0.5,
    presence_penalty: float = 0.2,
    stop: list = None,
    format: str = None,
) -> str:

    response: ChatResponse = chat(
        model=model,
        format=format,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
            "penalize_newline": True,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
        },
        stream=False,
    )

    return response.message.content


def translate(
    text: str,
    model: str,
    context: str,
    source_language: str,
    target_language: str = "English",
) -> str:
    # Normalize non-Japanese text early
    if not contains_japanese(text):
        return unicodedata.normalize("NFKC", text)

    # Main translation call
    user_prompt = get_formatted_user_prompt(
        context, text, source_language, target_language
    )
    response = call_llm(
        model, SYSTEM_PROMPT, user_prompt, format=Translation.model_json_schema()
    )
    translation = Translation.model_validate_json(response)

    cleaned_text = translation.translated_texts[0]
    cleaned_text = clean_translated_text(cleaned_text)

    # Debug prints (optional — comment out in production)
    print(f"Input: {text}")
    print(f"Output: {cleaned_text}\n")

    # Fallback if translation failed (contains trigger words or Japanese)
    if (
        "translat" in cleaned_text.lower()
        or "onomatopoeia" in cleaned_text.lower()
        or contains_japanese(cleaned_text)
    ):
        cleaned_text = fallback_translation(
            text, model, source_language, target_language
        )

    return cleaned_text


def fallback_translation(
    text: str, model: str, source_language: str, target_language: str
) -> str:
    """Fallback: act as Google Translate for direct Japanese → target translation."""
    system_prompt = f"Your role is to act as DeepL translate. Translate the given a text in{source_language} to {target_language}. Output ONLY the translation."
    user_prompt = f"Translate to {target_language}: {text}"

    fallback_translation = call_llm(model, system_prompt, user_prompt)
    return clean_translated_text(fallback_translation)
