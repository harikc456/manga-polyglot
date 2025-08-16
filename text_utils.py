import re
import jaconv
from ollama import chat
from ollama import ChatResponse

system_prompt = """You are a manga translator for one of the biggest publishers in the world. You will be given the entire context of a page or chapter and then will be asked to translate a specific text which is part of that context. You must use the context to provide an accurate translation of the text. Do not translate the context in one go. Ensure that the translated text remains meaningful taking the overall context into account. Ensure the tenses and proverbs are consistent across the translation. Use a casual tone and eliminate redundancy in the translation.

You should only translate the text given between the <text> tags. Do not provide any reasoning or reflection about the translation."""

def get_formatted_user_prompt(context: str, text: str, target_language: str):
    user_prompt = f"""Translate the given text from Japanese manga to {target_language}. Provide only the translation and nothing else.

    Context: The current page contains the following texts: <context>{context}</context>

    Use the context to provide the {target_language} translation for the Japanese text between <text> tags: <text>{text}</text>

    The response SHOULD be in {target_language}."""
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
        },
        stream=False,
    )
    text = response.message.content
    cleaned_text = re.sub(r"\n+", " ", text)
    return cleaned_text
