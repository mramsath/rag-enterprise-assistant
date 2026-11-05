
import streamlit as st

from ask import ask_question

st.set_page_config(
    page_title="Enterprise AI Assistant",
    page_icon="🤖"
)

st.title("Enterprise AI Knowledge Assistant")

st.caption(
    "Ask questions about your company documents."
)

question = st.chat_input("Ask something...")

if question:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            try:
                answer = ask_question(question)
                st.write(answer)

            except Exception as e:
                st.error(f"Error: {e}")
