import os
import sys

import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from loguru import logger

from Config import config
from llm.ChatCore import chat_with_history
from llm.RagCore import get_answer
from uicomponent.StComponent import side_bar_links


@st.cache_resource
def setup_log():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add('log/runtime_{time}.log', rotation='00:00', level='INFO', retention='10 days')
    logger.add('log/error_{time}.log', rotation='00:00', level='ERROR', retention='10 days')


st.set_page_config(
    page_title='学术大模型知识库',
    page_icon='📖',
    layout='wide',
    menu_items={
        'Report a bug': 'https://github.com/Aye10032/AcademyKnowledgeBot/issues',
        'About': 'https://github.com/Aye10032/AcademyKnowledgeBot'
    }
)

os.environ["LANGCHAIN_PROJECT"] = 'AcademyLLMChat'
setup_log()

milvus_cfg = config.milvus_config
col = milvus_cfg.COLLECTIONS
collections = []
for collection in col:
    collections.append(collection.NAME)

title = milvus_cfg.get_collection().TITLE
st.title(title)

with st.sidebar:
    side_bar_links()

    st.markdown('#### 选择知识库')
    option = st.selectbox('选择知识库',
                          range(len(collections)),
                          format_func=lambda x: collections[x],
                          label_visibility='collapsed')

    st.caption(f'当前数据库为：{milvus_cfg.get_collection().NAME}')

    if not option == milvus_cfg.DEFAULT_COLLECTION:
        config.set_collection(option)
        st.rerun()

    st.markdown('#### 选择对话模式')
    st.toggle('对话模式', key='chat_type', label_visibility='collapsed')

    if st.session_state.get('chat_type'):
        st.caption('当前为对话模式')
    else:
        st.caption('当前为知识库查询模式')

    st.markdown('#### 精准询问')
    st.toggle('精准询问', key='self_query', label_visibility='collapsed')

    if st.session_state.get('self_query'):
        st.caption('精准模式：:green[开]')
        with st.expander('索引条件'):
            st.text_input('doi', key='target_doi')
            st.text_input('title', key='target_title')
            st.text_input('author', key='target_author')
    else:
        st.caption('精准模式：:red[关]')

    st.divider()
    st.subheader('使用说明')
    st.markdown("""
    **:blue[知识库查询模式]**:  
    为单次对话请求，回答完全来自RAG返回的参考文献
    
    **:blue[对话模式]**  
    会以当前对话框中内容为基础开始GPT问答
    """)

prompt = st.chat_input('请输入问题')

col_chat, col_doc = st.columns([1, 1], gap='large')

with col_chat:
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    with st.container(height=610, border=False):
        for message in st.session_state.messages:
            icon = 'logo.png' if message['role'] != 'user' else None
            with st.chat_message(message['role'], avatar=icon):
                st.markdown(message['content'])

        if prompt:
            st.chat_message('user').markdown(prompt)

            # if st.session_state.get('chat_type'):
            #     logger.info(f'chat: {prompt}')
            #     st.session_state.messages.append({'role': 'user', 'content': prompt})
            #
            #     chat_history = ChatMessageHistory()
            #     for message in st.session_state.messages:
            #         if message['role'] == 'assistant':
            #             chat_history.add_ai_message(message['content'])
            #         else:
            #             chat_history.add_user_message(message['content'])
            #
            #     response = chat_with_history(chat_history, prompt)
            #
            #     st.chat_message('assistant', avatar='logo.png').markdown(response.content)
            #     st.session_state.messages.append({'role': 'assistant', 'content': response.content})
            #     logger.info(f'answer: {response.content}')

with col_doc:
    if 'documents' not in st.session_state:
        st.session_state.documents = []

    if 'cite_list' not in st.session_state:
        st.session_state.cite_list = []

    if len(st.session_state.documents) > 0:
        st.subheader('参考文献')
        with st.container(height=550, border=True):
            for index, ref in enumerate(st.session_state.documents):
                _title = ref.metadata['title']
                _author = ref.metadata['author']
                _year = ref.metadata['year']
                _doi = ref.metadata['doi']
                if index in st.session_state.get('cite_list'):
                    st.markdown(f'#### ✅{_title}')
                else:
                    st.markdown(f'#### {_title}')
                st.caption(f'{_author}({_year}) [{_doi}](https://doi.org/{_doi})')
                st.markdown(ref.page_content)
                st.divider()

if prompt:
    if not st.session_state.get('chat_type'):
        logger.info(f'question: {prompt}')
        st.session_state.messages = [{'role': 'user', 'content': prompt}]

        response = get_answer(prompt, st.session_state.get('self_query'))

        st.session_state.documents = response['docs']

        answer = response['answer'][0]
        st.session_state.cite_list = answer['citations']
        cite_str = ','.join(str(cit + 1) for cit in answer['citations'])
        answer_str = f"{answer['answer_en']}\n\n{answer['answer_zh']}\n\n参考文献：[{cite_str}]"
        st.session_state.messages.append({'role': 'assistant', 'content': answer_str})
        logger.info(f"answer: {response['answer'][0]['answer_zh']}")

        st.rerun()
    else:
        logger.info(f'chat: {prompt}')
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        chat_history = ChatMessageHistory()
        for message in st.session_state.messages:
            if message['role'] == 'assistant':
                chat_history.add_ai_message(message['content'])
            else:
                chat_history.add_user_message(message['content'])

        response = chat_with_history(chat_history, prompt)

        st.chat_message('assistant', avatar='logo.png').markdown(response.content)
        st.session_state.messages.append({'role': 'assistant', 'content': response.content})
        logger.info(f'answer: {response.content}')

        st.rerun()
