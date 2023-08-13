import os
from typing import List
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import tiktoken
from langchain.docstore.document import Document
import unicodedata
import pinecone
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
import pinecone

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def convert_filename_to_key(input_string):
    # Normalize string to decomposed form (separate characters and diacritics)
    normalized_string = unicodedata.normalize('NFKD', input_string)
    # Convert non-ASCII characters to ASCII
    ascii_string = normalized_string.encode('ascii', 'ignore').decode('utf-8')
    # Replace spaces, hyphens, and periods with _ underscores
    replaced_string = ascii_string.replace(' ', '_').replace('-', '_').replace('.', '_')
    return replaced_string

def get_all_filenames_and_their_extensions(source_folder):
    """  
    # Usage example
    source_folder = './raw_data/01 - Setor Imobiliario'

    files = get_all_filenames_and_their_extensions(source_folder)
    for file in files:
        print(f'File: {file[0]}, Extension: {file[1]}')
    """


    file_list = []  # Initialize an empty list to store the file information
    
    # Traverse through the source folder and its subfolders
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            file_name, file_extension = os.path.splitext(file_path)
            
            # Add the file information to the list as a tuple
            file_list.append((file_path, file_extension[1:].upper()))

    return file_list

def num_tokens_from_string(chunked_docs: List[Document]) -> int:

    string = ""
    print(f"Number of vectors: \n{len(chunked_docs)}")
    for doc in chunked_docs:
        string += doc.page_content

    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def update_vectorstore_PINECONE(docs: List[Document]):
    """Embeds a list of Documents and adds them to a Pinecone Index"""
    
    # Embed the chunks
    embeddings = OpenAIEmbeddings()  # type: ignore

    pinecone.init(api_key=os.environ["PINECONE_API_KEY"],environment=os.environ["PINECONE_ENVIRONMENT"])

    Pinecone.from_documents(docs,embeddings,index_name=os.environ["PINECONE_INDEX_NAME"],namespace="")

def update_vectorstore_FAISS(docs: List[Document]):
    """ Updates the FAISS index"""
    
    embeddings = OpenAIEmbeddings()  # type: ignore
    index = FAISS.from_documents(docs, embeddings)
    try:
        existing_index = FAISS.load_local("law_docs_index", embeddings)
        existing_index.merge_from(index)
        existing_index.save_local("law_docs_index")
    except Exception as e:
        print(f"Index doesn't exist. Starting fresh...")
        index.save_local("law_docs_index")

def remove_files_from_pinecone(path):
    
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"],environment=os.environ["PINECONE_ENVIRONMENT"])

    index = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])

    files = get_all_filenames_and_their_extensions(path)
    for file in files:
        filename_key = convert_filename_to_key(os.path.split(file[0])[-1])
        
        index.delete(
            filter={
                "filename_key": {"$eq": f"{filename_key}"},
            }
        )

        print(f"Removed file from Pinecone: {filename_key}")
        # index.describe_index_stats()