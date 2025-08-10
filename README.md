# manga-polyglot

## About

This project is a tool for translating manga images from one language to another. It uses a combination of text detection, optical character recognition (OCR), and a large language model (LLM) to perform the translation.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/manga-polyglot.git
    cd manga-polyglot
    ```

2.  **Install the dependencies:**

    This project uses Poetry for dependency management.

    ```bash
    pip install poetry
    poetry install
    ```

    Alternatively, you can install the dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```


## Configuration

The project uses a `config.json` file for configuration. Here's an explanation of the fields:

*   `text_detection_model_path`: Path to the text detection model.
*   `ocr_model`: The name of the OCR model to use from the Hugging Face Hub.
*   `llm_name`: The name of the large language model to use for translation.
*   `font_path`: Path to the font to use for rendering the translated text.

## Usage

To translate manga images, run the `inference.py` script:

```bash
python inference.py --input-dir <path_to_input_directory> --output-dir <path_to_output_directory>
```

*   `--input-dir`: The directory containing the images to be translated.
*   `--output-dir`: The directory where the translated images will be saved.

## Models

This project uses the following models:

*   **Text Detection:** `comictextdetector.pt` - A model for detecting text in comic images.
*   **OCR:** `kha-white/manga-ocr-base` - A model for recognizing text in manga images.
*   **LLM:** `qwen3:4b-instruct-2507-q4_K_M` - A large language model for translation.

## Acknowledgments

This project would not be possible without the work of the following open-source projects:

*   **[python-image-translator](https://github.com/boysugi20/python-image-translator) by [boysugi20](https://github.com/boysugi20):** This project provided the initial inspiration and a solid foundation for the image translation pipeline.
*   **[comic-text-detector](https://github.com/dmMaze/comic-text-detector) by [dmMaze](https://github.com/dmMaze):** The powerful text detection model from this repository is used to accurately locate text in manga images.
*   **[manga_ocr](https://github.com/kha-white/manga_ocr) by [kha-white](https://github.com/kha-white):** This project's OCR model is used to extract Japanese text from the images.

I am incredibly grateful to the authors and contributors of these projects for their valuable work and for making it available to the community.
