import streamlit as st

from rag.RagCore import ask_from_rag

st.set_page_config(page_title="微藻文献大模型知识库", page_icon="📖", layout='wide')
st.title('微藻文献大模型知识库')

with st.sidebar:
    st.header("RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(''):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = ask_from_rag(prompt)
    with st.chat_message("assistant"):
        st.markdown(response['result'])
        with st.expander('参考文献'):
            for ref in response['source_documents']:
                _title = ref.metadata['Title']
                st.markdown(f'#### {_title}')
                st.markdown(ref.page_content)
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response['result']})
