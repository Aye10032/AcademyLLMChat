import os
from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from storage.SqliteStore import ProfileStore
from uicomponent.StatusBus import *
from utils.entities.TimeZones import time_zone_list
from utils.entities.UserProfile import User, UserGroup, Project, ChatHistory

config = get_config()


def side_bar_links():
    st.header('欢迎使用学术LLM知识库')

    st.page_link('App.py', label='知识库问答', icon='💬')
    st.page_link('pages/WriteAssistant.py', label='写作助手', icon='✍️')
    st.page_link('pages/FileUpload.py', label='上传文件', icon='📂')
    st.page_link('pages/CollectionManage.py', label='系统管理', icon='⚙️')

    st.divider()


@st.dialog("登录")
def login():
    username = st.text_input('user name')
    password = st.text_input('password', type='password')

    if st.button("Submit"):
        with ProfileStore(
                connection_string=config.get_user_db()
        ) as profile_store:
            login_result, user = profile_store.valid_user(username, password)

        if login_result:
            st.session_state['user_role'] = user

            if user.user_group == UserGroup.FILE_ADMIN:
                set_admin_enable()
            elif user.user_group == UserGroup.ADMIN:
                set_owner_enable()

            st.rerun()
        else:
            st.error('密码错误！')


def login_message(wide=False):
    if wide:
        _, col_auth_2, _ = st.columns([1.5, 3, 1.5], gap='medium')
        auth_holder = col_auth_2.empty()
    else:
        auth_holder = st.empty()

    with auth_holder.container(border=True):
        st.warning('您无法使用本页面的功能，请登录相应权限的账户')
        st.button('login', on_click=lambda: login())


def score_text(score: float) -> str:
    red = int((1 - score) * 255)
    green = int(score * 255)
    blue = 0
    alpha = 200
    color_code = "#{:02x}{:02x}{:02x}{:02x}".format(red, green, blue, alpha)
    html_str = (f'<span style="display: inline-block; padding: 5px 5px; margin: 5px; '
                f'background-color: {color_code}; color: white; '
                f'border-radius: 10px; font-size: 10px; font-family: Arial, sans-serif;">{round(score, 4)}</span>')

    return html_str


@st.dialog('Create project')
def create_project(user: User):
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
                # TODO
                # 创建相关数据库等
                os.makedirs(
                    os.path.join(config.get_user_path(), user.name, project_name),
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
