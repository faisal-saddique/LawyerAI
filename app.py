# Import necessary libraries and modules
import chat_utils
import openai
from dotenv import load_dotenv
import os
import streamlit as st
from streaming import StreamHandler

from langchain.memory import StreamlitChatMessageHistory

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings  # For generating embeddings with OpenAI's embedding model
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables from .env file
load_dotenv()

# Set the title of the Streamlit app
st.title("LawyerAI - Chatbot 🤖")

# Check if the chat history is present in session state, if not, initialize it
if "session_chat_history" not in st.session_state:
    st.session_state.session_chat_history = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

if "index" not in st.session_state:
    try:
        pinecone.init(api_key=os.environ["PINECONE_API_KEY"],environment=os.environ["PINECONE_ENVIRONMENT"])
        pinecone_start = Pinecone.from_existing_index(os.environ["PINECONE_INDEX_NAME"], embeddings)
        st.session_state["index"] = pinecone_start.as_retriever(k=3)

    except Exception as e:
        st.error(f"FAISS index couldn't be loaded.\n\n{e}")

# Define a class for the custom chatbot
class CustomDataChatbot:

    def __init__(self):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def create_qa_chain(self):

        # Define the system message template
        system_template = """You are a helpful assistant. Always end your sentence asking your users if they need more help. Use the following pieces of context to answer the users question at the end. 
        If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer. If the question is a greeting or a goodbye, then be flexible and respond accordingly.
        ----------------
        {context}
        ----------------
        
        This is the history of your conversation with the user so far:
        ----------------
        {chat_history}
        ----------------"""

        # Create the chat prompt templates
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(
                "Question:```{question}```")
        ]

        qa_prompt = ChatPromptTemplate.from_messages(messages)

        # Optionally, specify your own session_state key for storing messages
        msgs = StreamlitChatMessageHistory(key="special_app_key")

        memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=msgs)

        # Create a faiss vector store using an existing index and OpenAI embeddings
        vectorstore = st.session_state.index

        # Create a ConversationalRetrievalChain for question answering
        st.session_state.qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(streaming=True,
                       temperature=0, model="gpt-3.5-turbo"),  # Chat model configuration
            # Use the faiss vector store for retrieval
            vectorstore,
            # Another OpenAI model for condensing questions
            condense_question_llm=OpenAI(temperature=0),
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            # memory=memory,  # Provide the conversation memory
            return_source_documents=True,
        )

    @chat_utils.enable_chat_history
    def main(self):

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            chat_utils.display_msg(user_query, 'user')

            # Display assistant's response with avatar
            with st.chat_message("assistant", avatar="https://icon-library.com/images/law-icon-png/law-icon-png-3.jpg"):
                st_callback = StreamHandler(st.empty())
                if 'qa' not in st.session_state:
                    st.session_state.qa = None
                    self.create_qa_chain()
                result = st.session_state.qa({"question": user_query, "chat_history": st.session_state.chat_history}, callbacks=[
                                             st_callback])
            

                # Display the source documents and metadata
                with st.expander("See sources"):
                    for doc in result['source_documents']:
                        st.info(f"\nPage Content: {doc.page_content}")
                        st.json(doc.metadata, expanded=False)

                st.session_state.messages.append(
                    {"role": "assistant", "content": result['answer'], "matching_docs": result['source_documents']})
                st.session_state.session_chat_history.append(
                    (user_query, result["answer"]))
                st.session_state.chat_history.append(
                    (user_query, result["answer"]))

# Entry point for the script
if __name__ == "__main__":
    if "index" in st.session_state:
        obj = CustomDataChatbot()
        obj.main()