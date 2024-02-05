import os
import time
from datetime import datetime

import streamlit as st
from tqdm import tqdm

import Config
from Config import config
from utils.GrobidUtil import parse_xml
from utils.MarkdownPraser import split_markdown
from utils.PubmedUtil import get_paper_info

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
        col1, col2 = st.columns([2, 1], gap='medium')
        with col1:
            st.markdown('选择知识库')
            option = st.selectbox('选择知识库',
                                  range(len(collections)),
                                  format_func=lambda x: collections[x],
                                  label_visibility='collapsed')

        with col2:
            st.markdown('选择文献所属年份')

            current_year = datetime.now().year
            year = st.selectbox('Year',
                                [year for year in range(1900, current_year + 1)][::-1],
                                label_visibility='collapsed')

        st.markdown(' ')

        st.markdown('上传markdown文件')
        uploaded_files = st.file_uploader(
            '上传Markdown文件',
            type=['md'],
            accept_multiple_files=True,
            label_visibility='collapsed')
        st.warning('请将markdown文件重命名为文献对应的doi号，并将doi号中的/替换为@，如10.1007@s00018-023-04986-3.md')

        submit = st.form_submit_button('导入文献', type='primary')

    if submit:
        file_count: int = len(uploaded_files)
        progress_text = f'正在处理文献(0/{file_count})，请勿关闭或刷新此页面'
        md_bar = st.progress(0, text=progress_text)
        for index, uploaded_file in tqdm(enumerate(uploaded_files), total=file_count):
            doc = split_markdown(uploaded_file, year)
            progress_num = (index + 1) / file_count
            time.sleep(1)
            md_bar.progress(progress_num, text=f'正在处理文本({index + 1}/{file_count})，请勿关闭或刷新此页面')
        md_bar.empty()
        st.write('处理完成，共添加了', file_count, '份文献')

with tab2:
    with st.form('pdf_form'):
        st.markdown('选择知识库')
        option = st.selectbox('选择知识库',
                              range(len(collections)),
                              format_func=lambda x: collections[x],
                              label_visibility='collapsed')
        st.warning('由于PDF解析需要请求PubMed信息，为了防止')
        uploaded_file = st.file_uploader('选择PDF文件', type=['pdf'])

        submit = st.form_submit_button('导入文献', type='primary')

    if submit:
        if uploaded_file is not None:
            # Read the PDF file
            # Extract the content
            data = get_paper_info(uploaded_file.name.replace('.pdf', '').replace('PM', ''))
            pdf_path = os.path.join(Config.get_work_path(),
                                    config.DATA_ROOT,
                                    milvus_cfg.COLLECTIONS[option].NAME,

                                    data['year'],
                                    uploaded_file.name)
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            with open(pdf_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            xml_path = os.path.join(Config.get_work_path(),
                                    config.DATA_ROOT,
                                    milvus_cfg.COLLECTIONS[option].NAME,
                                    data['year'],
                                    uploaded_file.name)

            result = parse_xml(config.get_xml_path() + '/2010/10.1016@j.biortech.2010.03.103.grobid.tei.xml')
            st.write(result)
