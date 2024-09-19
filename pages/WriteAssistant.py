import os
from datetime import datetime
from typing import Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_milvus import Milvus
from loguru import logger
from langchain_community.chat_message_histories import SQLChatMessageHistory
from streamlit.runtime.uploaded_file_manager import UploadedFile

import utils.MarkdownPraser as md
from Config import Config
from llm.ChatCore import write_paper, conclude_chat
from llm.GraphCore import write_with_db
from llm.ModelCore import load_embedding
from llm.RagCore import load_vectorstore, load_doc_store
from llm.RetrieverCore import insert_retriever
from storage.SqliteStore import ProfileStore
from uicomponent.StComponent import side_bar_links, login_message
from uicomponent.StatusBus import get_config, get_user, update_user
from utils.entities.TimeZones import time_zone_list
from utils.entities.UserProfile import Project, User, UserGroup, ChatHistory

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
collections = [collection.collection_name for collection in milvus_cfg.collections]

user: User = get_user()


@st.dialog('Create project')
def __create_project():
    project_name = st.text_input('Project name')
    time_zone = st.selectbox(
        'Time zone',
        index=285,
        options=time_zone_list
    )

    st.button(
        'Create',
        key='create_project',
        type='primary',
        disabled=not project_name
    )
    if st.session_state.get('create_project'):
        now_time = datetime.now().timestamp()

        project = Project(
            name=project_name,
            owner=user.name,
            last_chat=str(uuid4()),
            update_time=now_time,
            create_time=now_time,
            time_zone=time_zone,
        )
        with ProfileStore(
                connection_string=config.get_user_db()
        ) as profile_store:
            project_success = profile_store.create_project(project)

            if project_success:
                # 创建向量数据库
                embedding = load_embedding()
                index_param = {
                    "metric_type": "L2",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64}
                }
                collection_name = (
                    f"{project.owner}_"
                    f"{datetime.fromtimestamp(now_time, tz=ZoneInfo(time_zone)).strftime('%Y_%m_%d_%H_%M_%S')}"
                )
                vector_db = Milvus(
                    embedding,
                    collection_name=collection_name,
                    connection_args=config.milvus_config.get_conn_args(),
                    index_params=index_param,
                    drop_old=True,
                    auto_id=True,
                    enable_dynamic_field=True
                )
                init_doc = Document(
                    page_content=f'This is a collection about test',
                    metadata={
                        'title': 'About this collection',
                        'section': 'Abstract',
                        'author': 'administrator',
                        'year': datetime.now().year,
                        'type': -1,
                        'keywords': 'collection',
                        'is_main': True,
                        'doi': ''
                    }
                )

                result = vector_db.add_documents([init_doc])
                vector_db.delete(ids=result)

                # 创建本地文件夹
                os.makedirs(
                    os.path.join(config.get_user_path(), user.name, project_name, 'origin_file'),
                    exist_ok=True
                )
                os.makedirs(
                    os.path.join(config.get_user_path(), user.name, project_name, 'markdown'),
                    exist_ok=True
                )

                # 更新用户最新工程
                user.last_project = project.name
                update_user(user)
                st.session_state['now_project'] = project.name

                chat_history = ChatHistory(
                    session_id=project.last_chat,
                    description='new chat',
                    owner=project.owner,
                    project=project.name,
                    update_time=now_time,
                    create_time=now_time,
                )

                chat_success = profile_store.create_chat_history(chat_history)

                if chat_success:
                    st.session_state['now_chat'] = chat_history.session_id
                    st.rerun()
                else:
                    st.warning(f'Automatic creation of dialogues for project {project.owner}/{project.name} has failed, '
                               f'please create them manually later!')
            else:
                st.warning(f'Project {project.owner}/{project.name} already exist!')


def __change_project():
    user.last_project = st.session_state.get('now_project')
    update_user(user)


def __change_chat():
    now_time = datetime.now().timestamp()

    # 更新数据库
    project = Project(
        name=st.session_state.get('now_project'),
        owner=user.name,
        last_chat=st.session_state['now_chat'],
        update_time=now_time,
    )

    with ProfileStore(
            connection_string=config.get_user_db()
    ) as profile_store:
        profile_store.update_project(project)


def __on_create_chat_click():
    now_time = datetime.now().timestamp()

    chat = ChatHistory(
        session_id=str(uuid4()),
        description='新对话',
        owner=user.name,
        project=st.session_state.get('now_project'),
        update_time=now_time
    )
    project = Project(
        name=st.session_state.get('now_project'),
        owner=user.name,
        last_chat=chat.session_id,
        update_time=now_time,
    )

    with ProfileStore(
            connection_string=config.get_user_db()
    ) as profile_store:
        profile_store.create_chat_history(chat)
        profile_store.update_project(project)

    st.session_state['now_chat'] = chat.session_id


def __on_summary_click():
    chat_message_history = SQLChatMessageHistory(
        session_id=st.session_state.get('now_chat'),
        connection=f"sqlite:///{config.get_user_path()}/{user.name}/chat_history.db"
    )
    chat_description = conclude_chat(chat_message_history)

    now_time = datetime.now().timestamp()
    chat = ChatHistory(
        session_id=st.session_state.get('now_chat'),
        description=chat_description.content,
        owner=user.name,
        project=user.last_project,
        update_time=now_time,
    )
    project = Project(
        name=st.session_state.get('now_project'),
        owner=user.name,
        last_chat=st.session_state.get('now_chat'),
        update_time=now_time
    )
    with ProfileStore(
            connection_string=config.get_user_db()
    ) as profile_store:
        profile_store.update_project(project)
        profile_store.update_chat_history(chat)


def __on_file_upload(
        main_file: Optional[UploadedFile],
        other_files: Optional[UploadedFile | list[UploadedFile]]
):
    embedding = load_embedding()
    vector_db = load_vectorstore(st.session_state.get('now_main_collection'), embedding)
    doc_db = load_doc_store(
        os.path.join(
            config.get_user_path(),
            user.name,
            st.session_state.get('now_project'),
            'document.db'
        )
    )
    retriever = insert_retriever(vector_db, doc_db, 'zh')

    if main_file:
        match main_file.type:
            case 'md':
                NotImplementedError()
            case _:
                ...


def __main_page():
    chat_message_history = SQLChatMessageHistory(
        session_id=st.session_state.get('now_chat'),
        connection=f"sqlite:///{config.get_user_path()}/{user.name}/chat_history.db"
    )

    prompt = st.chat_input('请输入问题')

    col_chat, col_conf = st.columns([2, 1], gap='small')

    col_chat.caption(f'{user.name}/{user.last_project}')
    chat_container = col_chat.container(height=690, border=False)
    with chat_container:
        for message in chat_message_history.messages:
            with st.chat_message(message.type):
                st.markdown(message.content)

    config_container = col_conf.container(height=730, border=True)
    with config_container:
        with st.expander('##### 文件上传'):
            main_file = st.file_uploader(
                '主文件上传',
                type=['md']
            )
            other_files = st.file_uploader(
                '其他材料上传',
                type=['md'],
                accept_multiple_files=True,
            )
            st.button(
                '上传',
                on_click=__on_file_upload,
                args=[main_file, other_files]
            )

        st.divider()

        st.multiselect(
            '知识库调用',
            options=collections
        )

        st.divider()
        st.markdown('##### 常用功能')
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
        chat_message_history.add_user_message(HumanMessage(content=prompt))
        logger.info(f'({user.name}) chat: {prompt}')

        write_graph = write_with_db()
        response = write_graph.stream({"messages":chat_message_history.messages})

        result = chat_container.chat_message('assistant').write_stream(response)
        chat_message_history.add_ai_message(AIMessage(content=result))
        logger.info(f"(gpt4o-mini) answer: {result}")


def __different_ui():
    if user.user_group < UserGroup.WRITER or user.name == config.yml['user_login_config']['admin_user']['username']:
        with st.sidebar:
            side_bar_links()

        login_message(True)
    elif not user.last_project:
        with st.sidebar:
            side_bar_links()

        _, col_auth_2, _ = st.columns([1.5, 3, 1.5], gap='medium')
        with col_auth_2.container(border=True):
            st.warning('您目前还没有工程，请先创建一个工程以开始使用：')
            st.button(
                'Create',
                type='primary',
                on_click=__create_project
            )
    else:
        with ProfileStore(
                connection_string=config.get_user_db()
        ) as profile_store:
            project_list = [
                _project.name
                for _project in profile_store.get_project_list(user.name)
            ]
            chat_list = profile_store.get_chat_list(user.name, user.last_project)

            _last_project = profile_store.get_project(user.name, user.last_project)
            st.session_state['now_project'] = user.last_project
            st.session_state['now_main_collection'] = (
                f"{_last_project.owner}_"
                f"{datetime.fromtimestamp(_last_project.create_time, tz=ZoneInfo(_last_project.time_zone)).strftime('%Y_%m_%d_%H_%M_%S')}"
            )
            st.session_state['now_chat'] = _last_project.last_chat
            logger.info(f"Load chat({st.session_state['now_chat']}) from {user.name}/{st.session_state['now_project']}")

        with st.sidebar:
            side_bar_links()

            st.selectbox(
                'Project',
                options=project_list,
                key='now_project',
                on_change=__change_project,
            )
            st.button(
                '➕',
                key='btn_new_proj',
                on_click=__create_project,
                help='新建工程'
            )

            st.selectbox(
                '对话历史',
                options=chat_list.keys(),
                key='now_chat',
                format_func=lambda x: chat_list[x],
                on_change=__change_chat,
            )
            col_new_chat, col_summary, _ = st.columns([0.5, 1, 1])
            col_new_chat.button(
                '➕',
                key='btn_new_chat',
                on_click=__on_create_chat_click,
                help='新建对话'
            )
            col_summary.button(
                'Summary',
                on_click=__on_summary_click
            )

        __main_page()


st.header('AI写作助手')
__different_ui()
