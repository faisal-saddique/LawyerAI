from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import time
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from utilities.sidebar import sidebar

sidebar()

st.title("Ask Docs AI 🤖")

# Load environment variables from .env file
load_dotenv()

embeddings = OpenAIEmbeddings()

llm = ChatOpenAI(
    model_name=os.getenv('MODEL_NAME') or "gpt-3.5-turbo", # type: ignore
    temperature=os.getenv('MODEL_TEMPERATURE') or .3,
    max_tokens=os.getenv('MAX_TOKENS') or 1500
)

template = f"{os.getenv('OPENAI_GPT_INSTRUCTIONS')}" + '\n```{documents}```'

system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = '{question}'
human_message_prompt = HumanMessagePromptTemplate.from_template(
    human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])

# chain = LLMChain(llm=llm, prompt=chat_prompt)

if "index" in st.session_state:
    query = st.text_input('Enter a question: ')

    if query:
        start_time = time.perf_counter()

        st.write("Starting the search operation...")
        
        docs = st.session_state.index.similarity_search(query, k=3)

        # st.write(num_tokens_from_string(docs))  

        with st.expander("Show Matched Chunks"):
            for idx, doc in enumerate(docs):
                st.write(f"**Chunk # {idx+1}**")
                st.write(f"*{doc.page_content}*")
                st.json(doc.metadata, expanded=False)

        st.write("Found relevant docs, proceeding to make the query...")

        # You can uncomment the line below and disable the answer fetching from the other method to jump to chain method, which you were using initially
        # answer = chain.run(documents=docs, question=query)

        # This section uses chat completion models to generate the results. You can change the model to text-ada-001 or other models to compare the results from them
        # llm_light = OpenAI(model_name="text-davinci-003",
        #                     temperature=settings.OPENAI_TEMPERATURE, openai_api_key=settings.OPENAI_API_KEY)
        # answer = llm_light(
        # f"{settings.OPENAI_GPT_INSTRUCTIONS}/n{query}/n/nUse this context only:/n{docs}")

        # You can uncomment the line below and disable the answer fetching from the other method to jump to GPT-4 chat mode
        answer = llm(chat_prompt.format_prompt(documents=[doc.page_content for doc in docs], question=query).to_messages()).content

        end_time = time.perf_counter()
        st.success(answer)
        elapsed_time = end_time - start_time
        st.write(f"\nElapsed time: {round(float(elapsed_time), 3)} secs")

else:
    st.warning("Please create a knowledgeBase first!")
