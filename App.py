import os
import sys
from datetime import datetime

import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from loguru import logger
from pandas import DataFrame

from Config import Config
from llm.ChatCore import chat_with_history
from llm.RagCore import get_answer
from llm.RetrieverCore import get_expr
from uicomponent.StComponent import side_bar_links, score_text
from uicomponent.StatusBus import get_config, update_config


@st.cache_resource
def setup_log():
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(f'log/runtime_{year}{month:02d}{day:02d}.log', rotation='00:00', level='INFO', retention='10 days')
    logger.add(f'log/error_{year}{month:02d}{day:02d}.log', rotation='00:00', level='ERROR', retention='10 days')


st.set_page_config(
    page_title='学术大模型知识库',
    page_icon='📖',
    layout='wide',
    menu_items={
        'Report a bug': 'https://github.com/Aye10032/AcademyLLMChat/issues',
        'About': 'https://github.com/Aye10032/AcademyLLMChat'
    }
)

os.environ["LANGCHAIN_PROJECT"] = 'AcademyLLMChat'
setup_log()

config: Config = get_config()
milvus_cfg = config.milvus_config
col = milvus_cfg.collections
collections = []
for collection in col:
    collections.append(collection.collection_name)

title = milvus_cfg.get_collection().TITLE
st.title(title)

with st.sidebar:
    side_bar_links()

    st.markdown('#### 选择知识库')
    option = st.selectbox('选择知识库',
                          range(len(collections)),
                          format_func=lambda x: collections[x],
                          label_visibility='collapsed')

    if not option == milvus_cfg.default_collection:
        config.set_collection(option)
        update_config(config)
        st.caption(f'当前数据库为：{milvus_cfg.get_collection().NAME}')
    else:
        st.caption(f'当前数据库为：{milvus_cfg.get_collection().NAME}')

    st.divider()
    st.markdown('#### Advance')

    st.selectbox('选择LLM',
                 options=['gpt3.5', 'gpt3.5-16k', 'gpt4'],
                 index=1,
                 key='LLM')

    st.toggle('对话模式', key='chat_type')

    if st.session_state.get('chat_type'):
        st.caption('当前为对话模式')
    else:
        st.caption('当前为知识库查询模式')

    st.toggle('精准询问', key='self_query')

    if st.session_state.get('self_query'):
        st.caption('精准模式：:green[开]')
        with st.expander('索引条件'):
            st.text_input('doi', key='target_doi')
            st.text_input('title', key='target_title')
            st.text_input('year', key='target_year')
            st.text_input('author', key='target_author')
            st.toggle('模糊匹配', key='fuzzy_mode')
    else:
        st.caption('精准模式：:red[关]')

    st.toggle('双语回答', True, key='show_en')

    st.divider()
    st.subheader('使用说明')
    st.markdown("""
    **:blue[对话模式]**:  
    默认为查询模式，仅支持单次对话，根据RAG返回答案；若切换为对话模式，则根据现有聊天记录开始常规的LLM问答。
    
    **:blue[精准询问]**:    
    仅在查询模式下生效，默认情况下会使用LLM自动分析提问的自然语言，返回条件搜索结果。也可以手动指定查找文献的条件。
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

with col_doc:
    if 'documents' not in st.session_state:
        st.session_state.documents = []

    if 'cite_list' not in st.session_state:
        st.session_state.cite_list = []

    if len(st.session_state.documents) > 0:
        st.subheader('文献片段')
        with st.container(height=550, border=True):
            for index, ref in enumerate(st.session_state.documents):
                _title = ref.metadata['title']
                _author = ref.metadata['author']
                _year = ref.metadata['year']
                _doi = ref.metadata['doi']
                _score = ref.metadata['score']

                if index in st.session_state.get('cite_list'):
                    st.markdown(f'#### ✅【{index + 1}】{_title}')
                else:
                    st.markdown(f'#### 【{index + 1}】{_title}')
                st.caption(
                    f'{_author}({_year}) [{_doi}](https://doi.org/{_doi}) {score_text(_score)}',
                    unsafe_allow_html=True
                )

                main_content: str = str(ref.page_content)

                for _sentence in ref.metadata['refer_sentence']:
                    main_content = main_content.replace(_sentence, f' :orange[{_sentence}]')

                st.markdown(main_content)
                st.divider()

if prompt:
    if not st.session_state.get('chat_type'):
        collection_name = milvus_cfg.get_collection().NAME

        logger.info(f'question ({collection_name}): {prompt}')
        st.session_state.messages = [{'role': 'user', 'content': prompt}]

        if st.session_state.get('self_query'):
            kwarg: dict = {}
            if target_doi := st.session_state.get('target_doi'):
                kwarg['doi'] = target_doi

            if target_title := st.session_state.get('target_title'):
                kwarg['title'] = target_title

            if target_year := st.session_state.get('target_year'):
                kwarg['year'] = target_year

            if target_author := st.session_state.get('target_author'):
                kwarg['author'] = target_author

            if len(kwarg) != 0:
                expr = get_expr(st.session_state.get('fuzzy_mode'), **kwarg)
                response = get_answer(
                    collection_name,
                    prompt,
                    True,
                    expr,
                    llm_name=st.session_state.get('LLM')
                )
            else:
                response = get_answer(collection_name, prompt, True, llm_name=st.session_state.get('LLM'))
        else:
            response = get_answer(collection_name, prompt, llm_name=st.session_state.get('LLM'))

        st.session_state.documents = response['docs']

        answer = response['answer']

        cite_list = []
        cite_dict = {}
        for cite_id in answer['citations']:
            if 0 < cite_id <= len(response['docs']):
                cite_list.append(cite_id - 1)

                sub_doc = response['docs'][cite_id - 1]
                _title = sub_doc.metadata['title']
                _author = sub_doc.metadata['author']
                _year = sub_doc.metadata['year']
                _doi = sub_doc.metadata['doi']

                key = (_doi, _title, _author, _year)

                if key not in cite_dict:
                    cite_dict[key] = [str(cite_id)]
                else:
                    cite_dict[key].append(str(cite_id))

        cite_str_list = []
        for key in sorted(cite_dict, key=lambda x: cite_dict[x]):
            cite_str_list.append(
                f"[{','.join(cite_dict[key])}] \"{key[1]}\" {key[2]} ({key[3]}) [{key[0]}](https://doi.org/{key[0]})")

        st.session_state.cite_list = cite_list
        cite_str = '\n\n'.join(cite_str_list)

        if st.session_state.get('show_en'):
            answer_str = f"{answer['answer_en']}\n\n{answer['answer_zh']}\n\n**参考文献**: \n\n{cite_str}"
        else:
            answer_str = f"{answer['answer_zh']}\n\n**参考文献**: \n\n{cite_str}"
        st.session_state.messages.append({'role': 'assistant', 'content': answer_str})
        logger.info(f"({st.session_state.get('LLM')}) answer: {answer['answer_zh']}")

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
        logger.info(f"({st.session_state.get('LLM')}) answer: {response.content}")

        st.rerun()
