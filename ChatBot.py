import utilities.chat_utilities as chat_utilities
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from utilities.sidebar import sidebar
from utilities.streaming import StreamHandler
import pinecone
import openai
import tiktoken
import json

sidebar()

st.title("LawyerAI 🎓🤖")

# Load environment variables from .env file
load_dotenv()

embeddings = OpenAIEmbeddings()

class CustomDataChatbot:

    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        openai.api_key = os.getenv("OPENAI_API_KEY")

    # Function to count the number of tokens in a text string
    def num_tokens_from_string(self, string: str, encoding_name: str = "cl100k_base") -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    # @st.cache_resource
    def get_vectorstore_instance(self):

        pinecone.init(api_key=os.environ["PINECONE_API_KEY"],environment=os.environ["PINECONE_ENVIRONMENT"])
        vectordb = Pinecone.from_existing_index(os.environ["PINECONE_INDEX_NAME"], embeddings, namespace="notion_db")

        return vectordb

    # Function to manage chat history by adding messages and handling token limits
    def manage_chat_history(self, role, content):

        st.session_state.chat_messages.append({"role": f"{role}", "content": f"{content}"})

        # Count the number of tokens used in the chat history
        chat_history_tokens = self.num_tokens_from_string(json.dumps(st.session_state.chat_messages))
        print(f"\nChat history consumes {chat_history_tokens} tokens up till now\n")

        # Check if the token limit (3500 tokens) is about to be hit, if yes, remove extra messages
        if chat_history_tokens >= 3500:
            print("Avoiding token limit hit, removing extra chat messages...")
            # Keep only the system prompt and last 2 messages to reduce token usage
            messages = [st.session_state.chat_messages[0]] + st.session_state.chat_messages[-2:]

    # Function to get the AI assistant's response using OpenAI's ChatCompletion API
    def get_gpt_response(self, messages):
        # Make a request to the GPT-3.5 Turbo model to get the response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=1024,
            stream=True
        )

        # Extract and return the content of the AI assistant's response
        return response

    @st.spinner('Analyzing documents..')
    # Function to retrieve matching chunks from the vector store for a given query
    def get_matching_chunks_from_vecstore(self, vectorstore, query: str):

        # Perform similarity search in the vector store and get the top 3 most similar documents
        docs = vectorstore.similarity_search(query, k=4)

        with st.expander("Show Matched Chunks"):
            for idx, doc in enumerate(docs):
                st.write(f"**Chunk # {idx+1}**")
                st.text(f"{doc.page_content}")
                st.json(doc.metadata, expanded=False)

        # Prepare a formatted string with the content of each document
        context = "\n--------------------------------------\n"
        for doc in docs:
            context += doc.page_content + "\n--------------------------------------\n"

        return context

    # Function to create an input prompt for the chat with the AI assistant
    def create_input_prompt(self, vectorstore, query: str):

        # Get the relevant context based on the query from the vector store
        context = self.get_matching_chunks_from_vecstore(vectorstore=vectorstore, query=query)

        # Combine the context and query to form the input prompt for the AI assistant
        prompt = f"""Answer the question in your own words as truthfully as possible from the context given to you.\nIf you do not know the answer to the question, simply respond with "I don't know. Can you ask another question".\nIf questions are asked where there is no relevant context available, simply respond with "I don't know. Please ask a question relevant to the documents"\n\nCONTEXT: {context}\n\nHUMAN: {query}\nASSISTANT:"""

        return prompt
    
    @chat_utilities.enable_chat_history
    def main(self):

        user_query = st.chat_input(placeholder="Ask me anything!")

        # check 
        # Start the conversation by introducing the AI assistant
        self.manage_chat_history("system", "You are a helpful assistant.")

        if user_query:
            
            vectorstore = self.get_vectorstore_instance()

            chat_utilities.display_msg(user_query, 'user')

            with st.chat_message("assistant",avatar="https://icon-library.com/images/law-icon-png/law-icon-png-3.jpg"):
                
                query = self.create_input_prompt(vectorstore=vectorstore,query=user_query)

                # Add user's query to the chat history
                self.manage_chat_history("user", query)

                # Get the AI assistant's response
                response = self.get_gpt_response(messages=st.session_state.chat_messages)

                # def stream_response(response):
                st_cb = StreamHandler(st.empty())
                resp_for_chat_history = ""
                for chunk in response:
                    if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                        resp_for_chat_history += chunk['choices'][0]['delta']['content']
                        st_cb.on_llm_new_token(chunk['choices'][0]['delta']['content'])

                st.session_state.messages.append({"role": "assistant", "content": resp_for_chat_history})


if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()