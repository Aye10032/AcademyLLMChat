import os

import streamlit as st
from loguru import logger
from langchain_community.chat_message_histories import ChatMessageHistory

from Config import Config
from llm.ChatCore import write_paper
from uicomponent.StComponent import side_bar_links
from uicomponent.StatusBus import get_config

st.set_page_config(
    page_title="学术大模型知识库",
    page_icon="📖",
    layout='wide',
    menu_items={
        'Report a bug': 'https://github.com/Aye10032/AcademyLLMChat/issues',
        'About': 'https://github.com/Aye10032/AcademyLLMChat'
    }
)

os.environ["LANGCHAIN_PROJECT"] = 'WriteAssistant'
config: Config = get_config()
milvus_cfg = config.milvus_config
col = milvus_cfg.collections
collections = []
for collection in col:
    collections.append(collection.collection_name)

with st.sidebar:
    side_bar_links()

    st.selectbox(
        'Project',
        options=['测试工程2024']
    )
    st.button(
        '➕',
    )

    st.selectbox(
        '对话历史',
        options=['简述...']
    )
    st.button(
        '开始新对话',
    )

st.header('AI写作助手')

if 'write_messages' not in st.session_state:
    st.session_state.write_messages = []

prompt = st.chat_input('请输入问题')

col_chat, col_conf = st.columns([2, 1], gap='small')

col_chat.caption('aye/测试文档工程2024')
chat_container = col_chat.container(height=650, border=True)
with chat_container:
    for message in st.session_state.write_messages:
        icon = 'logo.png' if message['role'] != 'user' else None
        with st.chat_message(message['role'], avatar=icon):
            st.markdown(message['content'])

config_container = col_conf.container(height=690, border=True)
with config_container:
    st.file_uploader('主文件上传')
    st.file_uploader(
        '其他材料上传',
        accept_multiple_files=True
    )

    st.divider()

    st.multiselect(
        '知识库调用',
        options=collections
    )

    st.divider()
    st.subheader('常用功能')
    btn_col1, btn_col2, _, _ = st.columns([1, 1, 1, 1], gap='small')
    with btn_col1:
        st.button(
            '风格仿写',
            type='primary'
        )

    with btn_col2:
        st.button(
            '自动纠错',
            type='primary'
        )

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
    logger.info(f"(gpt4o-mini) answer: {result}")
