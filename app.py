import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from streamlit_extras.add_vertical_space import add_vertical_space
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import os
import pickle

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Sentence Transformers](https://www.sbert.net/)
    - [FAISS](https://github.com/facebookresearch/faiss)
    - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main)
    - [Distill Bert Model](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad)
 
    ''')
    add_vertical_space(5)
    st.write('By Nitya❤️  [Resume](https://nityaraji.github.io/My_Portfolio/)')
 
def main():
    st.header("Chat with PDF 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                index, embeddings_model = pickle.load(f)
            st.write('Embeddings Loaded from Disk')
        else:
            embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = embeddings_model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump((index, embeddings_model), f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            query_embedding = embeddings_model.encode([query])
            distances, indices = index.search(query_embedding, k=3)
            docs = [chunks[idx] for idx in indices[0]]
            
            # Use a transformer model for question answering
            qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
            best_answer = None
            highest_score = 0
            # answers = []
            for doc in docs:
                answer = qa_model(question=query, context=doc)
                if answer['score'] > highest_score:
                    highest_score = answer['score']
                    best_answer = answer
                # answers.append(answer)
            
            # Display the results
            # st.write("### Answers")
            # for answer in answers:
            #     st.write(f"Answer: {answer['answer']}")
            #     st.write(f"Score: {answer['score']}")
            #     st.write("---")

            # st.write(f"Answer: {answer['answer']}")
            # st.write(f"Score: {answer['score']}")
            # st.write("---")
            st.write("### Best Answer")
            st.write(f"Answer: {best_answer['answer']}")
            st.write(f"Score: {best_answer['score']}")

if __name__ == '__main__':
    main()
