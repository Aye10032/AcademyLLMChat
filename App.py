import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from loguru import logger

from Config import config
from llm.ChatCore import chat_with_history
from llm.RagCore import get_answer

logger.add('log/run_time.log')

milvus_cfg = config.milvus_config
col = milvus_cfg.COLLECTIONS
collections = []
for collection in col:
    collections.append(collection['collection_name'])

title = milvus_cfg.get_collection().DESCRIPTION
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

    st.markdown('#### 选择知识库')
    option = st.selectbox('选择知识库',
                          range(len(collections)),
                          format_func=lambda x: collections[x],
                          label_visibility='collapsed')

    if not option == milvus_cfg.DEFAULT_COLLECTION:
        config.set_collection(option)
        st.cache_resource.clear()

    st.caption(f'当前数据库为：{title}')

    st.markdown('#### 选择对话模式')
    chat_type = st.toggle('对话模式', label_visibility='collapsed')

    if chat_type:
        st.caption('当前为对话模式')
    else:
        st.caption('当前为知识库查询模式')

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
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        if prompt:
            st.chat_message('user').markdown(prompt)

            if chat_type:
                logger.info(f'chat: {prompt}')
                st.session_state.messages.append({'role': 'user', 'content': prompt})

                chat_history = ChatMessageHistory()
                for message in st.session_state.messages:
                    if message['role'] == 'assistant':
                        chat_history.add_ai_message(message['content'])
                    else:
                        chat_history.add_user_message(message['content'])

                response = chat_with_history(chat_history, prompt)

                st.chat_message('assistant').markdown(response.content)
                st.session_state.messages.append({'role': 'assistant', 'content': response.content})
                logger.info(f'answer: {response.content}')

with col_doc:
    if 'documents' not in st.session_state:
        st.session_state.documents = []

    if len(st.session_state.documents) > 0:
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
    if not chat_type:
        logger.info(f'question: {prompt}')
        st.session_state.messages = [{'role': 'user', 'content': prompt}]

        response = get_answer(prompt)

        st.session_state.documents = response['source_documents']
        st.session_state.messages.append({'role': 'assistant', 'content': response['result']})
        logger.info(f'answer: {response["result"]}')

        st.rerun()
