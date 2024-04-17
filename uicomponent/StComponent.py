from Config import UserRole
from uicomponent.StatusBus import *

config = get_config()


def side_bar_links():
    st.header('欢迎使用学术LLM知识库')

    st.page_link('App.py', label='知识库问答', icon='💬')
    st.page_link('pages/Translate.py', label='一键翻译总结', icon='🗎')
    st.page_link('pages/FileUpload.py', label='上传文件', icon='📂')
    st.page_link('pages/CollectionManage.py', label='知识库管理', icon='🖥️')

    st.divider()


def role_check(role: int, wide=False):
    if 'role' not in st.session_state:
        st.session_state['role'] = UserRole.VISITOR
        set_visitor_enable()

    if st.session_state.get('role') < role:
        if wide:
            _, col_auth_2, _ = st.columns([1.2, 3, 1.2], gap='medium')
            auth_holder = col_auth_2.empty()
        else:
            auth_holder = st.empty()

        with auth_holder.container(border=True):
            st.warning('您无法使用本页面的功能，请输入身份码')
            st.caption(f'当前的身份为{st.session_state.role}, 需要的权限为{UserRole.OWNER}')
            auth_code = st.text_input('身份码', type='password')

        if auth_code == config.ADMIN_TOKEN:
            st.session_state['role'] = UserRole.ADMIN
            set_admin_enable()
            auth_holder.empty()
        elif auth_code == config.OWNER_TOKEN:
            st.session_state['role'] = UserRole.OWNER
            set_owner_enable()
            auth_holder.empty()


def score_text(score: float) -> str:
    red = int((1 - score) * 255)
    green = int(score * 255)
    blue = 0
    alpha = 200
    color_code = "#{:02x}{:02x}{:02x}{:02x}".format(red, green, blue, alpha)
    html_str = f'<span style="display: inline-block; padding: 5px 5px; margin: 5px; background-color: {color_code}; color: white; border-radius: 10px; font-size: 10px; font-family: Arial, sans-serif;">{round(score, 4)}</span>'

    return html_str
