import os
import time
from datetime import datetime

import streamlit as st
from tqdm import tqdm

import Config
from Config import config, UserRole
from uicomponent.StComponent import side_bar_links
from utils.FileUtil import save_to_md
from utils.GrobidUtil import parse_xml, parse_pdf_to_xml
from utils.MarkdownPraser import split_markdown
from utils.PubmedUtil import get_paper_info

milvus_cfg = config.milvus_config
collections = []
for collection in milvus_cfg.COLLECTIONS:
    collections.append(collection.NAME)

st.set_page_config(page_title="学术大模型知识库", page_icon="📖", layout='wide')

with st.sidebar:
    side_bar_links()


if 'role' not in st.session_state:
    st.session_state['role'] = UserRole.VISITOR

if st.session_state.get('role') < UserRole.ADMIN:
    _, col_auth_2, _ = st.columns([1.2, 3, 1.2], gap='medium')
    auth_holder = col_auth_2.empty()
    with auth_holder.container(border=True):
        st.warning('您无法使用本页面的功能，请输入身份码')
        st.caption(f'当前的身份为{st.session_state.role}, 需要的权限为{UserRole.ADMIN}')
        auth_code = st.text_input('身份码', type='password')

    if auth_code == config.ADMIN_TOKEN:
        st.session_state['role'] = UserRole.ADMIN
        auth_holder.empty()
    elif auth_code == config.OWNER_TOKEN:
        st.session_state['role'] = UserRole.OWNER
        auth_holder.empty()


st.title('添加文献')


def markdown_tab():
    col_1, col_2, col_3 = st.columns([1.2, 3, 0.5], gap='medium')

    with col_1.container(border=True):
        st.subheader('使用说明')
        st.markdown(
            """   
            最为推荐的方式，不需要任何网络请求。先在本地手动将文献转为markdown文件之后再导入知识库，可以批量导入，但是每次仅能导入同一年的
            1. 将markdown文件重命名为文献对应的doi号，并将doi号中的`/`替换为`@`，如`10.1007@s00018-023-04986-3.md`          
            2. 选择文献所属年份     
            3. 上传markdown文件      
            4. 点击导入文献按钮，等待处理完成     
            """
        )

    with col_2.form('md_form'):
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


def pdf_tab():
    if 'md_text' not in st.session_state:
        st.session_state.md_text = ''

    if 'disable' not in st.session_state:
        st.session_state.disable = True

    col_1, col_2, col_3 = st.columns([1.2, 3, 0.5], gap='medium')

    with col_1.container(border=True):
        st.subheader('使用说明')
        st.warning('由于PDF解析需要请求PubMed信息，为了防止大量访问造成解析失败，仅允许上传单个文件')
        st.markdown(
            """  
            1. 将PDF文件重命名为`PMxxxx.pdf`的格式          
            2. 确保grobid已经在运行     
            3. 上传PDF文件      
            4. 等待处理完成     
            """
        )

    with col_2.form('pdf_form'):
        st.markdown('选择知识库')
        option = st.selectbox('选择知识库',
                              range(len(collections)),
                              format_func=lambda x: collections[x],
                              label_visibility='collapsed')
        uploaded_file = st.file_uploader('选择PDF文件', type=['pdf'])

        submit = st.form_submit_button('解析PDF')

        if submit:
            if uploaded_file is not None:
                data = get_paper_info(uploaded_file.name.replace('.pdf', '').replace('PM', ''))
                pdf_path = os.path.join(Config.get_work_path(),
                                        config.DATA_ROOT,
                                        milvus_cfg.COLLECTIONS[option].NAME,
                                        config.PDF_PATH,
                                        data['year'],
                                        uploaded_file.name)
                os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
                md_dict = [{'text': data['title'], 'level': 1},
                           {'text': 'Abstract', 'level': 2},
                           {'text': data['abstract'], 'level': 0}]

                with open(pdf_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                xml_path = os.path.join(Config.get_work_path(),
                                        config.DATA_ROOT,
                                        milvus_cfg.COLLECTIONS[option].NAME,
                                        config.XML_PATH,
                                        data['year'],
                                        data['doi'].replace('/', '@') + '.xml')

                os.makedirs(os.path.dirname(xml_path), exist_ok=True)
                with st.spinner('Parsing pdf...'):
                    _, _, xml_text = parse_pdf_to_xml(pdf_path)
                    with open(xml_path, 'w', encoding='utf-8') as f:
                        f.write(xml_text)
                    result = parse_xml(xml_path)

                    md_dict.extend(result['sections'])

                    md_path = os.path.join(Config.get_work_path(),
                                           config.DATA_ROOT,
                                           milvus_cfg.COLLECTIONS[option].NAME,
                                           config.MD_PATH,
                                           data['year'],
                                           data['doi'].replace('/', '@') + '.md')
                    save_to_md(md_dict, md_path)
                    with open(md_path, 'r', encoding='utf-8') as f:
                        md_text = f.read()

                    st.session_state.md_text = md_text
                    st.session_state.disable = False

                st.success('PDF识别完毕')

    st.markdown(' ')
    with st.container(border=True):
        md_col1, md_col2 = st.columns([1, 1], gap='medium')

        md_value = md_col1.text_area('文本内容', st.session_state.md_text, height=800, label_visibility='collapsed')

        if md_value:
            md_col2.container(height=800).write(md_value)

        submit = st.button('添加文献', type='primary', disabled=st.session_state.disable)


tab1, tab2, tab3 = st.tabs(['Markdown', 'PDF', 'Pubmed Center'])

with tab1:
    markdown_tab()

with tab2:
    pdf_tab()
