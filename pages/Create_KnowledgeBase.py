import streamlit as st
from utilities.utils import (
    refined_docs,
    parse_docx,
    parse_pdf,
    parse_xlsx,
    parse_csv,
    num_tokens_from_string,
    add_vectors_to_FAISS
)
from dotenv import load_dotenv

from utilities.sidebar import sidebar

sidebar()

# Load environment variables from .env file
load_dotenv()

# """ NOTE: Right now, only three formats of files are supported: PDF, DOCX & XLSX. """

st.title("Upload Documents")

accepted_file_types = ["pdf", "csv", "docx", "xlsx"]
uploaded_files = st.file_uploader("Upload one or more files", accept_multiple_files=True, type=accepted_file_types)

if uploaded_files:
    docs = None
    tot_len = 0

    for file in uploaded_files:
        file_extension = file.name.split(".")[-1].upper()
        st.write(f'File: {file.name}, Extension: {file_extension}')
        file_content = file.read()  # Read the content of the uploaded file

        # st.write(file_content)

        if file_extension == 'PDF':
            if docs is None:
                docs = parse_pdf(file_content)
            else:
                docs = docs + parse_pdf(file_content)
        elif file_extension == 'DOCX':
            if docs is None:
                docs = parse_docx(file_content)
            else:
                docs = docs + parse_docx(file_content)

        elif file_extension == 'XLSX':
            if docs is None:
                docs = parse_xlsx(file_content)
            else:
                docs = docs + parse_xlsx(file_content)
        
        elif file_extension == 'CSV':
            if docs is None:
                docs = parse_csv(file_content)
            else:
                docs = docs + parse_csv(file_content)
        else:
            raise ValueError("File type not supported!")
        
        # for doc in docs:
        #     st.success(doc)

    chunked_docs = refined_docs(docs)

    # for doc in chunked_docs:
    #     st.warning(doc)

    no_of_tokens = num_tokens_from_string(chunked_docs)
    st.write(f"Number of tokens: \n{no_of_tokens}")

    with st.spinner("Creating Index..."):
        st.session_state.index = add_vectors_to_FAISS(chunked_docs=chunked_docs)
        st.success("Done! Please headover to chatbot to start interacting with your data.")