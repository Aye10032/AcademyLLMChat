import streamlit as st

from Config import Config
from utils.entities.UserProfile import User, UserGroup


def set_visitor_enable():
    # file upload
    st.session_state['md_uploader_disable'] = True
    st.session_state['pdf_uploader_disable'] = True
    st.session_state['pmc_uploader_disable'] = True

    # collection manage
    st.session_state['manage_collection_disable'] = True
    st.session_state['new_collection_disable'] = True
    st.session_state['manage_user_disable'] = True


def set_admin_enable():
    # file upload
    st.session_state['md_uploader_disable'] = False
    st.session_state['pdf_uploader_disable'] = False
    st.session_state['pmc_uploader_disable'] = False

    # collection manage
    st.session_state['manage_collection_disable'] = True
    st.session_state['new_collection_disable'] = True
    st.session_state['manage_user_disable'] = True

    config = get_config()
    config.milvus_config.set_group_visibility(True)
    update_config(config)


def set_owner_enable():
    # file upload
    st.session_state['md_uploader_disable'] = False
    st.session_state['pdf_uploader_disable'] = False
    st.session_state['pmc_uploader_disable'] = False

    # collection manage
    st.session_state['manage_collection_disable'] = False
    st.session_state['new_collection_disable'] = False
    st.session_state['manage_user_disable'] = False

    # update visible collection
    config = get_config()
    config.milvus_config.set_group_visibility(True)
    update_config(config)


def get_config() -> Config:
    if 'config' not in st.session_state:
        st.session_state['config'] = Config()

    config: Config = st.session_state.get('config')
    return config


def update_config(config: Config) -> None:
    st.session_state['config'] = config

    milvus_cfg = config.milvus_config
    if milvus_cfg.get_collection().language == 'zh':
        st.session_state['app_is_zh_collection'] = True
    else:
        st.session_state['app_is_zh_collection'] = False


def get_user() -> User:
    if 'user_role' not in st.session_state:
        st.session_state['user_role'] = User(
            name=f'visitor',
            password='',
            user_group=UserGroup.VISITOR,
            last_project=''
        )

        set_visitor_enable()

    user: User = st.session_state.get('user_role')
    return user
