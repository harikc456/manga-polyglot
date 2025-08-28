import re
import jaconv
import base64
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
- The onomatopoeia needs to be translated appropriately
- Keep emoji's and english texts in the given text as it is while translating

Only translate the text given between the <text> tags. Your response should be concise and complete."""


def get_formatted_user_prompt(context: str, page_context: str, text: str, target_language: str):
    user_prompt = f"""Translate this Japanese manga text to {target_language}.

    Context: <context>{context}</context>

    Text to translate: <text>{text}</text>

    Provide the complete translation in {target_language}. Do not repeat words or phrases."""
    return user_prompt


def post_process(text):
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text


# def encode_image(image_path):
#   with open(image_path, "rb") as image_file:
#     return base64.b64encode(image_file.read()).decode('utf-8')

# def get_page_context(img_path: str) -> str:
#     return ""
#     system_prompt = """
#     You will be given a page from a manga, you have to scan through the page and give a descriptive context about what happens in the page.
#     This description is extremely essential to help the translator. The translator does not have access to the page and only has access to the text
#     in the page. So the page description given by you will be used by the Translator agent to generate accurate translations.
#     Generate the description in less than 500 words.
#     """

#     base64_img = encode_image(img_path)
#     response = chat(
#         model='gemma3',
#         messages=[
#             {
#             "role": "system",
#             "content": system_prompt,
#             },
#             {
#                 'role': 'user',
#                 'content': 'Generate the description on the events happening in the given page of manga',
#                 'images': [base64_img],
#             },
#         ],
#     )
#     print(response['message']['content'])
#     return response.message.content


def translate(text: str, model: str, context: str, page_context: str, target_language: str = "English") -> str:
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": get_formatted_user_prompt(context, page_context, text, target_language),
            },
        ],
        options={
            "temperature": 0.0,
            "num_ctx": 128,
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
