import streamlit as st

#decorator
def enable_chat_history(func):
    # to clear chat history after swtching chatbot
    current_page = func.__qualname__
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page
    if st.session_state["current_page"] != current_page:
        try:
            st.cache_resource.clear()
            del st.session_state["current_page"]
            del st.session_state["messages"]
        except:
            pass

    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    for msg in st.session_state["messages"]:
        if msg["role"] == "assistant":
            st.chat_message(msg["role"],avatar="https://icon-library.com/images/law-icon-png/law-icon-png-3.jpg").write(msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)