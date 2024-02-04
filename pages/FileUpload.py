import os

import pandas as pd
import streamlit as st
from st_milvus_connection import MilvusConnection

from Config import config

milvus_cfg = config.milvus_config
collections = []
for collection in milvus_cfg.COLLECTIONS:
    collections.append(collection.NAME)

st.set_page_config(page_title="微藻文献大模型知识库", page_icon="📖", layout='centered')
st.title('添加文献')

with st.sidebar:
    st.header('欢迎使用学术LLM知识库')

    st.page_link('App.py', label='Home', icon='💬')
    st.page_link('pages/FileUpload.py', label='上传文件', icon='📂')
    st.page_link('pages/CollectionManage.py', label='知识库管理', icon='🖥️')

    st.divider()

    st.title('使用说明')
    st.subheader('PDF')
    st.markdown(
        """由于学术论文的PDF中排版和图表的干扰，预处理较为复杂，建议尽量先在本地处理为markdown文件后再上传    
        1. 将PDF文件重命名为`PMxxxx.pdf`的格式          
        2. 确保grobid已经在运行     
        3. 上传PDF文件      
        4. 等待处理完成     
        """
    )
    st.subheader('Markdown')
    st.markdown(
        """   
        1. 将PDF文件重命名为`doi编号.md`的格式，并将doi编号中的`/`替换为`@`          
        2. 选择文献所属年份     
        3. 上传markdown文件      
        4. 等待处理完成     
        """
    )

tab1, tab2, tab3 = st.tabs(['Markdown', 'PDF', 'Pubmed Center'])

with tab1:
    with st.form('md_form'):
        st.subheader('选择知识库')

        option = st.selectbox('选择知识库',
                              range(len(collections)),
                              format_func=lambda x: collections[x],
                              label_visibility='collapsed')

        if not option == milvus_cfg.DEFAULT_COLLECTION:
            config.set_collection(option)
            st.cache_resource.clear()

        uploaded_files = st.file_uploader('选择Markdown文件', type=['md'], accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            # string_data = stringio.read()
            # st.markdown(string_data)

        st.form_submit_button('Submit my picks')

with tab2:
    uploaded_files = st.file_uploader('选择PDF文件', type=['pdf'], accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # string_data = stringio.read()
        # st.markdown(string_data)
