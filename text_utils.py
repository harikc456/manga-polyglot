import re
import jaconv
from ollama import chat
from ollama import ChatResponse


def post_process(text):
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text


def translate(text: str, model: str, context: str = "") -> str:
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """
                You are manga translator for one of the biggest publisher in the world, you will be given the entire context of page or
                chapter and then will be asked to translate a text which is part of the context. You have to use the context to give 
                accurate translate of the text. Do not translate the context in one go. Ensure that the translated text remains meaningful
                taking the overall context into account. Ensure the tenses and proverbs are consistent across the translation. 
                Use casual tone and eliminate redudancy in the translation
                
                You should only translate the text given between the <text> tags.
            """,
            },
            {
                "role": "user",
                "content": f"""Translate the given text from Japanese manga to English. Provide only the translation and nothing else.
            Context the current page contains the following texts <context> {context} </context>, Use the context to
            Provide the English translation for the text in Japanese between <text> tags <text>{text}</text>""",
            },
        ],
        options={
            "temperature": 0.0,
        },
        stream=False,
    )
    text = response.message.content
    cleaned_text = re.sub(r"\n+", " ", text)
    return cleaned_text
