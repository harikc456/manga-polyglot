import re
import jaconv
import unicodedata
from ollama import chat, ChatResponse

# Fixed System Prompt
SYSTEM_PROMPT = """
You are a manga translator for one of the biggest publishers in the world. 
You will be given the entire context of a page or chapter and then will be asked to translate a specific text which is part of that context. 
You must use the context to provide an accurate translation of the text. 
Do not translate the context in one go. Ensure that the translated text remains meaningful taking the overall context into account.
Ensure the tenses and proverbs are consistent across the translation.

CRITICAL OUTPUT RULES:
- Output ONLY the translated text, nothing else
- DO NOT include explanations, notes, or commentary about the translation
- The onomatopoeia needs to be translated according to how it sounds, you may use romanji here 
- Similarly translate, voices for suprises like for example え？ being "Eh?" or "Huh"?
- If the given text is a single unfinished word that you can't make of, then return the romanji
- Keep emoji's in the given text as it is while translating
- Use appropriate tone for translation as understood from the context
- The translated text is going to replace the original text, hence the length of the translation SHOULD be close the number of characters inside <text> tags
- Only translate the text given between the <text> tags. Your response should be concise and complete.
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


def get_formatted_user_prompt(context: str, text: str, target_language: str) -> str:
    return f"""Translate this Japanese manga text to {target_language}.

Context: <context>{context}</context>

Text to translate: <text>{text}</text>

The translation for {text} in {target_language} is:
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
    """Post-process Japanese text: normalize spacing, ellipses, and half-width chars."""
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
    num_ctx: int = 128,
    frequency_penalty: float = 0.5,
    presence_penalty: float = 0.2,
    stop: list = None,
) -> str:
    if stop is None:
        stop = ["\n\n", "---", "Note:"]

    response: ChatResponse = chat(
        model=model,
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
    text: str, model: str, context: str, target_language: str = "English"
) -> str:
    # Normalize non-Japanese text early
    if not contains_japanese(text):
        return unicodedata.normalize("NFKC", text)

    # Main translation call
    user_prompt = get_formatted_user_prompt(context, text, target_language)
    cleaned_text = call_llm(model, SYSTEM_PROMPT, user_prompt)
    cleaned_text = clean_translated_text(cleaned_text)

    # Debug prints (optional — comment out in production)
    # print(f"Input: {text}")
    # print(f"Output: {cleaned_text}\n")

    # Fallback if translation failed (contains trigger words or Japanese)
    if (
        "translat" in cleaned_text.lower()
        or "onomatopoeia" in cleaned_text.lower()
        or contains_japanese(cleaned_text)
    ):
        cleaned_text = fallback_translation(text, model, target_language)

    return cleaned_text


def fallback_translation(text: str, model: str, target_language: str) -> str:
    """Fallback: act as Google Translate for direct Japanese → target translation."""
    system_prompt = f"Your role is to act as google translate. Translate the given Japanese text into {target_language}. Output ONLY the translation."
    user_prompt = f"Translate to {target_language}: {text}"

    fallback_translation = call_llm(model, system_prompt, user_prompt)
    return clean_translated_text(fallback_translation)
