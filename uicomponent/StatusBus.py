import streamlit as st


def set_visitor_enable():
    # file upload
    st.session_state['md_uploader_disable'] = True
    st.session_state['pdf_uploader_disable'] = True
    st.session_state['pmc_uploader_disable'] = True

    # collection manage
    st.session_state['verify_text_disable'] = True
    st.session_state['new_collection_disable'] = True


def set_admin_enable():
    # file upload
    st.session_state['md_uploader_disable'] = False
    st.session_state['pdf_uploader_disable'] = False
    st.session_state['pmc_uploader_disable'] = False

    # collection manage
    st.session_state['verify_text_disable'] = True
    st.session_state['new_collection_disable'] = True


def set_owner_enable():
    # file upload
    st.session_state['md_uploader_disable'] = False
    st.session_state['pdf_uploader_disable'] = False
    st.session_state['pmc_uploader_disable'] = False

    # collection manage
    st.session_state['verify_text_disable'] = False
    st.session_state['new_collection_disable'] = False
