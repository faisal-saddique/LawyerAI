import pinecone
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings  

from dotenv import load_dotenv

load_dotenv()



# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),   # Find at app.pinecone.io
    environment=os.getenv("PINECONE_ENVIRONMENT"),   # Next to api key in console
)
# 
# index = pinecone.Index(os.getenv("PINECONE_INDEX"))
print(os.getenv("PINECONE_INDEX_NAME"))
print(os.getenv("PINECONE_API_KEY"))
print(os.getenv("PINECONE_ENVIRONMENT"))

# print(index.describe_index_stats())
# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()


pinecone_start = Pinecone.from_existing_index(os.environ["PINECONE_INDEX_NAME"], embeddings)


result = pinecone_start.similarity_search("law")

print(result)