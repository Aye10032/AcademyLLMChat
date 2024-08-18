import streamlit as st
from loguru import logger
from langchain_community.chat_message_histories import ChatMessageHistory

from llm.ChatCore import write_paper
from uicomponent.StComponent import side_bar_links

st.set_page_config(
    page_title="学术大模型知识库",
    page_icon="📖",
    layout='wide',
    menu_items={
        'Report a bug': 'https://github.com/Aye10032/AcademyLLMChat/issues',
        'About': 'https://github.com/Aye10032/AcademyLLMChat'
    }
)

with st.sidebar:
    side_bar_links()

st.title('AI写作助手')

if 'write_messages' not in st.session_state:
    st.session_state.write_messages = []

prompt = st.chat_input('请输入问题')

col_chat, col_doc = st.columns([2, 1], gap='large')

chat_container = col_chat.container(height=610, border=False)
with chat_container:
    for message in st.session_state.write_messages:
        icon = 'logo.png' if message['role'] != 'user' else None
        with st.chat_message(message['role'], avatar=icon):
            st.markdown(message['content'])

if prompt:
    chat_container.chat_message('user').markdown(prompt)
    logger.info(f'chat: {prompt}')
    st.session_state.write_messages.append({'role': 'user', 'content': prompt})

    chat_history = ChatMessageHistory()
    for message in st.session_state.write_messages:
        if message['role'] == 'assistant':
            chat_history.add_ai_message(message['content'])
        else:
            chat_history.add_user_message(message['content'])

    response = write_paper(chat_history, prompt)

    result = chat_container.chat_message('assistant', avatar='logo.png').write_stream(response)
    st.session_state.write_messages.append({'role': 'assistant', 'content': result})
    logger.info(f"({st.session_state.get('LLM')}) answer: {result}")
