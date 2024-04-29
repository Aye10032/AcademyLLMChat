import streamlit as st

from Config import Config


def set_visitor_enable():
    # file upload
    st.session_state['md_uploader_disable'] = True
    st.session_state['pdf_uploader_disable'] = True
    st.session_state['pmc_uploader_disable'] = True

    # collection manage
    st.session_state['manage_collection_disable'] = True
    st.session_state['new_collection_disable'] = True


def set_admin_enable():
    # file upload
    st.session_state['md_uploader_disable'] = False
    st.session_state['pdf_uploader_disable'] = False
    st.session_state['pmc_uploader_disable'] = False

    # collection manage
    st.session_state['manage_collection_disable'] = True
    st.session_state['new_collection_disable'] = True


def set_owner_enable():
    # file upload
    st.session_state['md_uploader_disable'] = False
    st.session_state['pdf_uploader_disable'] = False
    st.session_state['pmc_uploader_disable'] = False

    # collection manage
    st.session_state['manage_collection_disable'] = False
    st.session_state['new_collection_disable'] = False


def get_config() -> Config:
    if 'config' not in st.session_state:
        st.session_state['config'] = Config()

    config: Config = st.session_state.get('config')
    return config


def update_config(config: Config) -> None:
    st.session_state['config'] = config
