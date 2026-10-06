import os
import re
import jaconv
import unicodedata
from PIL import Image
from ollama import chat, ChatResponse
from pydantic import ValidationError
from data_model import Translation, SessionMemory
from memory_utils import serialize_memory, format_memory_for_prompt, format_glossary, filter_memory, merge_memory

# Room for the system prompt, surrounding-page context, session memory and previous translations;
# Ollama silently drops the start of prompts that exceed num_ctx.
TRANSLATE_NUM_CTX = 8192
# The memory update sees the known names, the summary and a whole page of originals and translations.
MEMORY_NUM_CTX = 8192

# Plain-text replies are wrapped in these tags: the reply is prefilled with the opening tag and
# generation stops at the closing one, so lead-ins ("Here is the translation:") and trailing notes
# never make it into the result, whatever their wording.
OPEN_TAG, CLOSE_TAG = "<translation>", "</translation>"
TAG_INSTRUCTION = f"Write the translation inside {OPEN_TAG}{CLOSE_TAG} tags, e.g. {OPEN_TAG}Want some coffee?{CLOSE_TAG}"

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

# Punctuation that carries tone (emphasis, hesitation) and must survive cleaning, even in runs.
_EXPRESSIVE_PUNCT = "!?！？…‥.。、・~〜"


# In vertical text the wave dash 〜 is drawn rotated and looks like an S, and OCR reads it as one (喂〜! -> 喂S!).
# An S counts as a misread only when it directly follows CJK text and is followed by the end, a space,
# punctuation or a small kana, which is where 〜 goes; words like Sランク, S級 or SNS are left alone.
_MISREAD_WAVE_DASH = re.compile(
    r"(?<=[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF])[SsＳｓ]+"
    r"(?=$|\s|[!?！？…‥.。、・~〜」』）)]|[っッぁぃぅぇぉァィゥェォ])"
)


def fix_misread_wave_dash(text: str) -> str:
    return _MISREAD_WAVE_DASH.sub(lambda m: "〜" * len(m.group()), text)


def clean_ocr_garbage(text: str) -> str:
    if not text.strip():
        return text

    text = fix_misread_wave_dash(text)

    # Remove long chains of identical box-drawing / line chars
    text = re.sub(r'([┌┐└┘├┤┬┴┼─│━┃═║]{3,})', '', text)          # ≥3 repeated line chars

    # Remove chains of almost any non-letter/symbol punctuation junk (expressive punctuation excepted)
    text = re.sub(rf'([^\w\s\u3000-\u30FF\u4E00-\u9FFF{re.escape(_EXPRESSIVE_PUNCT)}]{{3,}})', '', text)

    # Optional: collapse multiple punctuation → single (…… → …)
    text = re.sub(r'([…ー〜\-=]{2,})', lambda m: m.group(1)[0]*min(3, len(m.group(1))), text)

    # Remove trailing/leading junk that often appears. Not ー or …: they are part of the text.
    text = text.strip(' └│─═║┌┐┘├┤\u200b\ufeff')  # zero-width & invisible too

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


def is_japanese_source(source_language: str) -> bool:
    return source_language.strip().lower() in ("japanese", "ja", "jp", "日本語")


def has_letters(text: str) -> bool:
    """Return True if text contains any letter, in any script (not just punctuation/digits)."""
    return any(unicodedata.category(char).startswith("L") for char in text)


# Model output that talks about the translation instead of being one, e.g. "Here is the translation: ...".
_META_COMMENTARY = re.compile(r"^(here(?:'s| is)\b[^:]*translat|translated text\b|the translation (?:is|of)\b)")


def needs_fallback(translated: str, source_language: str) -> bool:
    """True when the model's output is unusable: empty, commentary, or still in the (Japanese) source script."""
    if not translated:
        return True
    lower = translated.lower()
    if _META_COMMENTARY.match(lower) or "onomatopoeia" in lower:
        return True
    return is_japanese_source(source_language) and contains_japanese(translated)


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
    - input_text - the original text that was supposed to be translated. The value of this field should be {text}.
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
    - input_text - the original text that was supposed to be translated which would be {text}. SHOULD NOT BE EMPTY
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
    prompt += f"Output ONLY the translated text. No explanation, no commentary.\n{TAG_INSTRUCTION}"
    return prompt


def extract_tagged(response: str) -> str:
    """Return the text between the translation tags; tolerates an echoed opening tag or an ignored stop."""
    if response.lstrip().startswith(OPEN_TAG):
        response = response.lstrip()[len(OPEN_TAG):]
    return response.split(CLOSE_TAG, 1)[0]


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
    prefill: str = None,
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

    if prefill:
        # Ollama continues a trailing assistant message; the reply contains only the continuation.
        messages.append({"role": "assistant", "content": prefill})

    response: ChatResponse = chat(
        model=model,
        format=format,
        messages=messages,
        # A top-level argument: inside options Ollama ignores it, and thinking models (e.g. qwen3.5)
        # then spend the whole reply on hidden reasoning and return empty content.
        think=False,
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
    image: Image.Image = None,
    previous_translations: list = None,
    session_memory: SessionMemory = None,
    use_json: bool = True,
) -> str:
    # Text with nothing to translate (no source-script characters / no letters) is only normalized
    japanese = is_japanese_source(source_language)
    if not (contains_japanese(text) if japanese else has_letters(text)):
        return unicodedata.normalize("NFKC", text)

    text = clean_ocr_garbage(text)
    if japanese:
        text = post_process(text)
    context = fix_misread_wave_dash(context)

    # Only the entities that occur around this bubble; the full glossary is mostly noise to the model.
    if session_memory is not None:
        session_memory = filter_memory(session_memory, f"{context}\n{text}")

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
            format=Translation.model_json_schema(), num_ctx=TRANSLATE_NUM_CTX, image=image,
        )
        try:
            translation = Translation.model_validate_json(response)
            cleaned_text = clean_translated_text(translation.translated_text)
        except ValidationError as e:
            print(f"[translate] WARNING: Invalid JSON response ({e}); using fallback translation.")
            cleaned_text = ""
    else:
        user_prompt = get_formatted_user_prompt_plain(
            context, text, source_language, target_language,
            previous_translations, session_memory=session_memory,
        )
        response = call_llm(
            model, SYSTEM_PROMPT, user_prompt, num_ctx=TRANSLATE_NUM_CTX, image=image,
            stop=[CLOSE_TAG], prefill=OPEN_TAG,
        )
        cleaned_text = clean_translated_text(extract_tagged(response))

    if needs_fallback(cleaned_text, source_language):
        cleaned_text = fallback_translation(
            text, model, source_language, target_language, image, session_memory=session_memory
        )

    print(f"Input: {text}")
    print(f"Output: {cleaned_text}\n")

    return cleaned_text


def fallback_translation(
    text: str,
    model: str,
    source_language: str,
    target_language: str,
    image: Image.Image = None,
    session_memory: SessionMemory = None,
) -> str:
    """Fallback: context-free, machine-translation-style prompt for when the main prompt's output is unusable.

    Only the glossary of known names is passed on, so names stay consistent with the main translations.
    """
    system_prompt = f"Your role is to act as DeepL translate. Translate the given text in {source_language} to {target_language}. Output ONLY the translation."
    user_prompt = f"Translate to {target_language}: {text}\n"
    glossary = format_glossary(session_memory) if session_memory else []
    if glossary:
        user_prompt += "Translate these names as given:\n" + "\n".join(f"- {line}" for line in glossary) + "\n"
    user_prompt += TAG_INSTRUCTION

    fallback_translation = call_llm(
        model, system_prompt, user_prompt, num_ctx=4096, image=image,
        stop=[CLOSE_TAG], prefill=OPEN_TAG,
    )
    return clean_translated_text(extract_tagged(fallback_translation))


MEMORY_UPDATE_SYSTEM_PROMPT = """
You are a manga translation assistant responsible for maintaining a translation memory.
Given the current page's translations and the names already known, you must:
1. List only NEW named entities from this page: characters (with gender inferred from speech/context/names), places, organizations.
   Copy each original name exactly as it is written in the original text, and its translation exactly as it is written in the translated text.
   Only proper names: no pronouns, titles on their own, common nouns or sound effects.
   Leave honorifics (先輩, さん, くん, ちゃん, 様) off the original name.
   Notes are a few words about the character (e.g. "protagonist", "Tanaka's sister"), or empty.
   Take only the name itself, never the particles or words around it. Example: from 「佐藤くんか…渋谷の店で待ってる」 /
   "Sato, huh... I'll wait at the shop in Shibuya." the entities are 佐藤 → Sato and 渋谷 → Shibuya
   (not 佐藤くんか, not 渋谷の店 → "the shop in Shibuya").
2. Do not repeat known entities, except a known character whose gender was unknown and is now clear from this page.
3. Rewrite the story summary to include events from this page (150 words max, cumulative)

Output ONLY valid JSON matching the provided schema. No commentary or explanation.
""".strip()


def update_session_memory(
    translations: list[dict],
    session_memory: SessionMemory,
    model: str,
    temp_dir: str,
) -> SessionMemory:
    """Ask the model for this page's new entities and summary, then merge them into memory in code.

    Existing entries are never dropped or renamed, and entities whose original name is not on the page are rejected,
    so a bad reply can at worst miss a new name.
    """
    if not translations:
        return session_memory

    known_names = [c.original_name for c in session_memory.characters]
    known_names += [e.original for e in session_memory.places + session_memory.organizations]
    unknown_gender = [c.original_name for c in session_memory.characters if c.gender == "unknown"]

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
    user_prompt = f"""Known names (do not repeat): {', '.join(known_names) if known_names else 'none'}
Known characters with unknown gender: {', '.join(unknown_gender) if unknown_gender else 'none'}
Story summary so far: {session_memory.story_summary or 'none'}

Current page translations:
{page_text}

Expected JSON shape: {schema_hint}
Return the new entities and the updated story summary as JSON."""

    response = call_llm(
        model,
        MEMORY_UPDATE_SYSTEM_PROMPT,
        user_prompt,
        format=SessionMemory.model_json_schema(),
        num_ctx=MEMORY_NUM_CTX,
        # Names must be copied exactly; repetition penalties push the model to alter or drop them.
        temperature=0.1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    try:
        update = SessionMemory.model_validate_json(response)
    except (ValueError, ValidationError) as e:
        print(f"[memory] WARNING: Failed to parse LLM memory update response: {e}. Keeping existing memory.")
        update = SessionMemory()

    source_text = "\n".join(t["original"] for t in translations)
    updated = merge_memory(session_memory, update, source_text)

    serialize_memory(updated, os.path.join(temp_dir, "memory.md"))
    return updated
