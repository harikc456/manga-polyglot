import os
import re
import jaconv
import unicodedata
from PIL import Image
from ollama import chat, ChatResponse
from pydantic import ValidationError
from data_model import Translation, SessionMemory
from memory_utils import serialize_memory, format_memory_for_prompt

# Fixed System Prompt
SYSTEM_PROMPT = """
You are an experienced manga translator for one of the biggest publishers in the world.
You will be given the context of surrounding pages and then asked to translate a specific text from the current page.

ATMOSPHERE & TONE (most important):
- Read the context carefully to infer the genre, mood, and setting (e.g. tense action, lighthearted comedy, emotional romance, horror, slice-of-life).
- Match your word choices and sentence rhythm to that inferred atmosphere. A battle cry should feel urgent and punchy. Casual banter should feel relaxed and colloquial. Heartfelt dialogue should feel warm and natural.
- Infer each character's speech register from context (formal, rough, childlike, archaic, etc.) and reflect it consistently.
- Prioritise natural, genre-appropriate dialogue over word-for-word literal accuracy. The translation should sound like something a real person would say in that genre — not a textbook sentence.

CONSISTENCY:
- Ensure tenses and pronouns are consistent across the translation.
- If previous translations are provided, use them as reference for terminology, character voice, and style.

CRITICAL OUTPUT RULES:
- DO NOT include explanations, breakdown, notes, or commentary about the translation
- Translate onomatopoeia according to how it sounds in the target language
- Translate surprise/reaction sounds naturally (e.g. え？ → "Huh?" or "Eh?", depending on the scene's tone)
- The translated text will replace the original, so its length SHOULD be close to the number of characters inside <text> tags
""".strip()

def clean_ocr_garbage(text: str) -> str:
    if not text.strip():
        return text

    # Remove long chains of identical box-drawing / line chars
    text = re.sub(r'([┌┐└┘├┤┬┴┼─│━┃═║]{3,})', '', text)          # ≥3 repeated line chars

    # Remove chains of almost any non-letter/symbol punctuation junk
    text = re.sub(r'([^\w\s\u3000-\u30FF\u4E00-\u9FFF]{3,})', '', text)

    # Optional: collapse multiple punctuation → single (…… → …)
    text = re.sub(r'([…ー〜\-=]{2,})', lambda m: m.group(1)[0]*min(3, len(m.group(1))), text)

    # Remove trailing/leading junk that often appears
    text = text.strip(' .,…ー〜└│─═║┌┐┘├┤\u200b\ufeff')  # zero-width & invisible too

    return text.strip()


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


def _build_prompt_base(
    context: str,
    text: str,
    source_language: str,
    target_language: str,
    previous_translations: list,
    session_memory: "SessionMemory | None",
) -> str:
    prompt = (
        f"Translate this {source_language} text from the manga to {target_language}.\n\n"
        f"Context: <context>{context}</context>\n\n"
        f"Text to translate: <text>{text}</text>\n\n"
    )
    memory_section = format_memory_for_prompt(session_memory) if session_memory else ""
    if memory_section:
        prompt += memory_section + "\n\n"
    if previous_translations:
        prompt += "Previous translations on this page (for consistency):\n"
        for i, prev in enumerate(previous_translations, 1):
            prompt += f"{i}. {source_language}: {prev['original']}\n   {target_language}: {prev['translated']}\n"
        prompt += "\n"
    return prompt


def get_formatted_user_prompt(
    context: str,
    text: str,
    source_language: str,
    target_language: str,
    previous_translations: list = None,
    session_memory: SessionMemory = None,
) -> str:
    prompt = _build_prompt_base(
        context, text, source_language, target_language,
        previous_translations, session_memory,
    )
    prompt += f"""Match the tone and atmosphere of the surrounding context in your translation.

- You are to return JSON structure output with two fields
    - text - the original text that was supposed to be translated. The value of this field should be {text}.
    - translated_text - the translation for the input text in {target_language}."""
    return prompt


def get_formatted_user_prompt_with_image(
    context: str,
    text: str,
    source_language: str,
    target_language: str,
    previous_translations: list = None,
    session_memory: SessionMemory = None,
) -> str:
    prompt = _build_prompt_base(
        context, text, source_language, target_language,
        previous_translations, session_memory,
    )
    prompt += f"""Match the tone and atmosphere of the surrounding context in your translation.

- You are to return JSON structure output with two fields
    - text - the original text that was supposed to be translated which would be {text}. SHOULD NOT BE EMPTY
    - translated_text - the translation for the input text in {target_language}.
- Use the provided image to aid your translation"""
    return prompt


def get_formatted_user_prompt_plain(
    context: str,
    text: str,
    source_language: str,
    target_language: str,
    previous_translations: list = None,
    session_memory: "SessionMemory | None" = None,
) -> str:
    prompt = _build_prompt_base(
        context, text, source_language, target_language,
        previous_translations, session_memory,
    )
    prompt += "Output ONLY the translated text. No explanation, no commentary."
    return prompt


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
    temperature: float = 0.25,
    num_ctx: int = 256,
    frequency_penalty: float = 0.5,
    presence_penalty: float = 1.5,
    stop: list = None,
    format: str = None,
    image: Image.Image = None,
) -> str:

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    if image is not None:
        # Convert PIL Image to bytes for ollama
        import io
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        
        messages.append({
            "role": "user",
            "content": user_prompt,
            "images": [image_bytes.getvalue()],
        })
    else:
        messages.append({"role": "user", "content": user_prompt})

    response: ChatResponse = chat(
        model=model,
        format=format,
        messages=messages,
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
            "penalize_newline": True,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "think": False, 
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
    image: Image.Image = None,
    previous_translations: list = None,
    session_memory: SessionMemory = None,
    use_json: bool = True,
) -> str:
    # Normalize non-Japanese text early
    if not contains_japanese(text):
        return unicodedata.normalize("NFKC", text)

    text = clean_ocr_garbage(text)
    text = post_process(text)

    if use_json:
        if image is not None:
            print("Using image for translation...")
            user_prompt = get_formatted_user_prompt_with_image(
                context, text, source_language, target_language,
                previous_translations, session_memory=session_memory,
            )
        else:
            user_prompt = get_formatted_user_prompt(
                context, text, source_language, target_language,
                previous_translations, session_memory=session_memory,
            )
        response = call_llm(
            model, SYSTEM_PROMPT, user_prompt,
            format=Translation.model_json_schema(), num_ctx=2048, image=image,
        )
        translation = Translation.model_validate_json(response)
        cleaned_text = clean_translated_text(translation.translated_text)

        if (
            "translat" in cleaned_text.lower()
            or "onomatopoeia" in cleaned_text.lower()
            or contains_japanese(cleaned_text)
        ):
            cleaned_text = fallback_translation(
                text, model, source_language, target_language, image
            )
    else:
        user_prompt = get_formatted_user_prompt_plain(
            context, text, source_language, target_language,
            previous_translations, session_memory=session_memory,
        )
        response = call_llm(
            model, SYSTEM_PROMPT, user_prompt, num_ctx=2048, image=image,
        )
        cleaned_text = clean_translated_text(response)

    print(f"Input: {text}")
    print(f"Output: {cleaned_text}\n")

    return cleaned_text


def fallback_translation(
    text: str, model: str, source_language: str, target_language: str, image: Image.Image = None
) -> str:
    """Fallback: act as Google Translate for direct Japanese → target translation."""
    system_prompt = f"Your role is to act as DeepL translate. Translate the given a text in{source_language} to {target_language}. Output ONLY the translation."
    user_prompt = f"Translate to {target_language}: {text}"

    fallback_translation = call_llm(model, system_prompt, user_prompt, num_ctx=4096, image=image)
    return clean_translated_text(fallback_translation)


MEMORY_UPDATE_SYSTEM_PROMPT = """
You are a manga translation assistant responsible for maintaining a translation memory.
Given the current page's translations and the existing memory state, you must:
1. Identify new named entities: characters (with gender inferred from speech/context/names), places, organizations
2. Preserve ALL entries from the existing memory — never drop an entity just because it does not appear in the current page. Only update an entry if a clear correction is warranted.
3. Rewrite the story summary to include events from this page (150 words max, cumulative)

Output ONLY valid JSON matching the provided schema. No commentary or explanation.
""".strip()


def update_session_memory(
    translations: list[dict],
    session_memory: SessionMemory,
    model: str,
    temp_dir: str,
) -> SessionMemory:
    if not translations:
        return session_memory

    current_chars = [
        f"{c.original_name} → {c.translated_name} ({c.gender})" +
        (f", {c.notes}" if c.notes else "")
        for c in session_memory.characters
    ]
    current_places = [f"{p.original} → {p.translated}" for p in session_memory.places]
    current_orgs = [f"{o.original} → {o.translated}" for o in session_memory.organizations]

    page_text = "\n".join(
        f"Original: {t['original']}\nTranslated: {t['translated']}"
        for t in translations
    )

    schema_hint = (
        '{"characters":[{"original_name":"...","translated_name":"...","gender":"male|female|unknown","notes":"..."}],'
        '"places":[{"original":"...","translated":"..."}],'
        '"organizations":[{"original":"...","translated":"..."}],'
        '"story_summary":"..."}'
    )
    user_prompt = f"""Current memory state:
Characters: {', '.join(current_chars) if current_chars else 'none'}
Places: {', '.join(current_places) if current_places else 'none'}
Organizations: {', '.join(current_orgs) if current_orgs else 'none'}
Story summary: {session_memory.story_summary or 'none'}

Current page translations:
{page_text}

Expected JSON shape: {schema_hint}
Return the complete updated memory as JSON."""

    response = call_llm(
        model,
        MEMORY_UPDATE_SYSTEM_PROMPT,
        user_prompt,
        format=SessionMemory.model_json_schema(),
        num_ctx=4096,
    )

    try:
        updated = SessionMemory.model_validate_json(response)
    except (ValueError, ValidationError) as e:
        print(f"[memory] WARNING: Failed to parse LLM memory update response: {e}. Keeping existing memory.")
        updated = session_memory

    serialize_memory(updated, os.path.join(temp_dir, "memory.md"))
    return updated
