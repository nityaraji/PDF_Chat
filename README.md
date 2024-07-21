# 🤗💬 LLM Chat App

This app is an LLM-powered chatbot that allows users to interact with PDF documents. It's built using the following technologies:

- [Streamlit](https://streamlit.io/): An open-source app framework for Machine Learning and Data Science teams.
- [Sentence Transformers](https://www.sbert.net/): Pre-trained models for efficient and scalable sentence and text embeddings.
- [FAISS](https://github.com/facebookresearch/faiss): A library for efficient similarity search and clustering of dense vectors.
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main): A Sentence Transformers model for generating embeddings.
- [Distill Bert Model](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad): A smaller, faster, cheaper, and lighter version of BERT for question answering.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/nityaraji/PDF_Chat.git
    cd llm-chat-app
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    source .venv/bin/activate  # On macOS/Linux
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Upload a PDF file using the file uploader in the sidebar.

3. Ask questions about the content of the PDF and get answers using the integrated language model.

## Example

![Screenshot](https://raw.githubusercontent.com/nityaraji/PDF_Chat/master/1.png)

## Author

Nitya❤️ [Resume](https://nityaraji.github.io/My_Portfolio/)
