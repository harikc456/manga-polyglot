import re
import jaconv
from ollama import chat
from ollama import ChatResponse

# Fixed System Prompt
system_prompt = """You are a manga translator for one of the biggest publishers in the world. You will be given the entire context of a page or chapter and then will be asked to translate a specific text which is part of that context. You must use the context to provide an accurate translation of the text. Do not translate the context in one go. Ensure that the translated text remains meaningful taking the overall context into account. Ensure the tenses and proverbs are consistent across the translation. Use a casual tone and eliminate redundancy in the translation.

CRITICAL OUTPUT RULES:
- Output ONLY the translated text, nothing else
- Do not repeat words or phrases multiple times
- Provide a single, complete translation
- Stop after completing the translation
- Do not include explanations, notes, or commentary
- Do not repeat any part of the translation

Only translate the text given between the <text> tags. Your response should be concise and complete."""


# system_prompt = """You are a manga translator for one of the biggest publishers in the world. You will be given the entire context of a page or chapter and then will be asked to translate a specific text which is part of that context. You must use the context to provide an accurate translation of the text. Do not translate the context in one go. Ensure that the translated text remains meaningful taking the overall context into account. Ensure the tenses and proverbs are consistent across the translation. Use a casual tone and eliminate redundancy in the translation.

# You should only translate the text given between the <text> tags. Do not provide any reasoning or reflection about the translation."""


def get_formatted_user_prompt(context: str, text: str, target_language: str):
    user_prompt = f"""Translate this Japanese manga text to {target_language}.

    Context: <context>{context}</context>

    Text to translate: <text>{text}</text>

    Provide one complete translation in {target_language}. Do not repeat words or phrases."""
    return user_prompt


def post_process(text):
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text


def translate(text: str, model: str, context: str = "", target_language: str = "English") -> str:
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": get_formatted_user_prompt(context, text, target_language),
            },
        ],
        options={
            "temperature": 0.0,
            "num_ctx": 50,
            "penalize_newline": True,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.2,
            # "repeat_penalty": 1.1,
            "stop": ["\n\n", "---", "Note:", "Translation:"],
        },
        stream=False,
    )
    text = response.message.content
    cleaned_text = re.sub(r"\n+", " ", text)
    return cleaned_text
