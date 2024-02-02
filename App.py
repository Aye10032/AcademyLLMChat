import streamlit as st
from loguru import logger

from Config import config
from llm.RagCore import get_answer

logger.add('log/run_time.log')

milvus_cfg = config.milvus_config
col = milvus_cfg.COLLECTIONS
collections = []
for collection in col:
    collections.append(collection['collection_name'])

title = milvus_cfg.get_collection()['description']
st.set_page_config(
    page_title='学术大模型知识库',
    page_icon='📖',
    layout='wide',
    menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': 'https://github.com/Aye10032/AcademyKnowledgeBot/issues',
        'About': 'https://github.com/Aye10032/AcademyKnowledgeBot'
    }
)
st.title(title)

with st.sidebar:
    st.header('欢迎使用学术LLM知识库')

    st.page_link('App.py', label='Home', icon='💬')
    st.page_link('pages/FileUpload.py', label='上传文件', icon='📂')

    st.divider()

    option = st.selectbox('选择知识库', range(len(collections)), format_func=lambda x: collections[x])

    chat_type = st.toggle('对话模式')

prompt = st.chat_input('请输入问题')

col_chat, col_doc = st.columns([1, 1], gap='large')

with col_chat:
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    with st.container(height=610, border=False):
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        if prompt:
            st.chat_message('user').markdown(prompt)

with col_doc:
    if 'documents' not in st.session_state:
        st.session_state.documents = []

    st.subheader('参考文献')
    with st.container(height=550, border=True):
        for ref in st.session_state.documents:
            st.divider()
            _title = ref.metadata['Title']
            _year = ref.metadata['year']
            _doi = ref.metadata['doi']
            st.markdown(f'#### {_title}')
            st.caption(f'{_doi} ({_year})')
            st.markdown(ref.page_content)

if prompt:
    st.session_state.messages = [{'role': 'user', 'content': prompt}]

    response = get_answer(prompt)

    st.session_state.documents = response['source_documents']
    st.session_state.messages.append({'role': 'assistant', 'content': response['result']})

    st.rerun()

if not option == milvus_cfg.DEFAULT_COLLECTION:
    config.set_collection(option)
    st.cache_resource.clear()
