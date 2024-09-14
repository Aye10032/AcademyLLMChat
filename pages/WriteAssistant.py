import os
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from loguru import logger
from langchain_community.chat_message_histories import ChatMessageHistory, SQLChatMessageHistory

from Config import Config
from llm.ChatCore import write_paper
from storage.SqliteStore import ProfileStore
from uicomponent.StComponent import side_bar_links, login_message
from uicomponent.StatusBus import get_config, get_user, update_user
from utils.entities.TimeZones import time_zone_list
from utils.entities.UserProfile import Project, User, UserGroup

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
def create_project():
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
        tz = ZoneInfo(time_zone)

        project = Project(
            name=project_name,
            owner=user.name,
            last_chat=datetime.fromtimestamp(now_time, tz).strftime("%Y-%m-%d %H:%M:%S"),
            create_time=now_time,
            update_time=now_time,
            archived=False
        )
        with ProfileStore(
                connection_string=config.get_user_db()
        ) as profile_store:
            result = profile_store.create_project(project)

        if result:
            # TODO
            # 创建相关数据库等

            # 更新用户最新工程
            user.last_project = project.name
            update_user(user)
            st.session_state['now_project'] = project.name

            st.rerun()
        else:
            st.warning(f'Project {project.owner}/{project.name} already exist!')


def __main_page():
    chat_message_history = SQLChatMessageHistory(
        session_id="test_session_id",
        connection="sqlite:///sqlite.db"
    )

    prompt = st.chat_input('请输入问题')

    col_chat, col_conf = st.columns([2, 1], gap='small')

    col_chat.caption(f'{user.name}/{user.last_project}')
    chat_container = col_chat.container(height=650, border=True)
    with chat_container:
        for message in chat_message_history.messages:
            icon = 'logo.png' if message.type != 'user' else None
            with st.chat_message(message.type, avatar=icon):
                st.markdown(message.content)

    config_container = col_conf.container(height=690, border=True)
    with config_container:
        with st.expander('#### 文件上传'):
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
                on_click=lambda: create_project()
            )
    else:
        with ProfileStore(
                connection_string=config.get_user_db()
        ) as profile_store:
            project_list = [
                _project.name
                for _project in profile_store.get_user_projects(user.name)
            ]

        with st.sidebar:
            side_bar_links()

            st.selectbox(
                'Project',
                key='now_project',
                options=project_list
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

        __main_page()


st.header('AI写作助手')
__different_ui()
