from storage.SqliteStore import ProfileStore
from uicomponent.StatusBus import *
from utils.entities.UserProfile import User, UserGroup

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


def role_check(role: int, wide=False):
    if 'user_role' not in st.session_state:
        st.session_state['user_role'] = User(
            name=f'visitor',
            password='',
            user_group=UserGroup.VISITOR,
            last_project=''
        )
        set_visitor_enable()

    user: User = st.session_state.get('user_role')
    if user.user_group < role:
        if wide:
            _, col_auth_2, _ = st.columns([1.2, 3, 1.2], gap='medium')
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
