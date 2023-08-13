import streamlit as st

description = """
**LawyerAI** is an AI-powered question-answering bot built using OpenAI's GPT models and FAISS/Pinecone vector search. It leverages language embeddings and vector stores to provide efficient and accurate search results.

To perform the search operation, **LawyerAI** employs either a FAISS index loaded from the local storage or a Pinecone Index loaded from cloud storage, which is created using OpenAI's embeddings. It uses the index to find the most relevant document chunks based on user queries. The retrieved document chunks are then displayed to the user, along with their metadata.

Once the relevant documents are obtained, **LawyerAI** uses OpenAI's GPT model to generate answers to user questions. It takes the retrieved documents and the user's query as input to the chat model and generates a response accordingly.

To use **LawyerAI**, simply enter your question in the provided text input field, and the bot will retrieve relevant document chunks and generate an answer based on the given query.
"""

def sidebar():
    with st.sidebar:
        st.title("About")
        st.write(f"{description}")