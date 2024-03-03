import os
from datetime import datetime

import streamlit as st
from tqdm import tqdm

import Config
from Config import config, UserRole
from llm.RagCore import load_vectorstore, load_doc_store
from llm.RetrieverCore import base_retriever
from uicomponent.StComponent import side_bar_links, role_check
from utils.FileUtil import save_to_md, section_to_documents
from utils.GrobidUtil import parse_xml, parse_pdf_to_xml
from utils.MarkdownPraser import split_markdown, split_markdown_text
from utils.PMCUtil import download_paper_data, parse_paper_data
from utils.PubmedUtil import get_paper_info

milvus_cfg = config.milvus_config
collections = []
for collection in milvus_cfg.COLLECTIONS:
    collections.append(collection.NAME)

st.set_page_config(page_title="学术大模型知识库", page_icon="📖", layout='wide')

with st.sidebar:
    side_bar_links()

role_check(UserRole.ADMIN, True)

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
                                  disabled=st.session_state['md_uploader_disable'],
                                  label_visibility='collapsed')

        with col2:
            st.markdown('选择文献所属年份')

            current_year = datetime.now().year
            year = st.selectbox('Year',
                                [year for year in range(1900, current_year + 1)][::-1],
                                disabled=st.session_state['md_uploader_disable'],
                                label_visibility='collapsed')

        st.markdown(' ')

        st.markdown('上传markdown文件')
        uploaded_files = st.file_uploader(
            '上传Markdown文件',
            type=['md'],
            accept_multiple_files=True,
            disabled=st.session_state['md_uploader_disable'],
            label_visibility='collapsed')

        submit = st.form_submit_button('导入文献',
                                       type='primary',
                                       disabled=st.session_state['md_uploader_disable'], )

        if submit:
            file_count = len(uploaded_files)
            if file_count == 0:
                st.warning('还没有上传文件')
                st.stop()

            config.set_collection(option)
            st.cache_resource.clear()
            vector_db = load_vectorstore()
            progress_text = f'正在处理文献(0/{file_count})，请勿关闭或刷新此页面'
            md_bar = st.progress(0, text=progress_text)
            for index, uploaded_file in tqdm(enumerate(uploaded_files), total=file_count):
                doc = split_markdown(uploaded_file, year)
                vector_db.add_documents(doc)
                progress_num = (index + 1) / file_count
                md_bar.progress(progress_num, text=f'正在处理文本({index + 1}/{file_count})，请勿关闭或刷新此页面')
            md_bar.empty()
            st.write('处理完成，共添加了', file_count, '份文献')


def pdf_tab():
    if 'md_text' not in st.session_state:
        st.session_state['md_text'] = ''

    if 'pdf_md_submit_disable' not in st.session_state:
        st.session_state['pdf_md_submit_disable'] = True

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
                              disabled=st.session_state['pdf_uploader_disable'],
                              label_visibility='collapsed')
        uploaded_file = st.file_uploader('选择PDF文件', type=['pdf'], disabled=st.session_state['pdf_uploader_disable'])

        submit = st.form_submit_button('解析PDF', disabled=st.session_state['pdf_uploader_disable'])

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

                doi = data['doi']
                year = data['year']
                st.session_state['paper_info'] = {'doi': doi, 'year': int(year)}

                xml_path = os.path.join(Config.get_work_path(),
                                        config.DATA_ROOT,
                                        milvus_cfg.COLLECTIONS[option].NAME,
                                        config.XML_PATH,
                                        year,
                                        doi.replace('/', '@') + '.xml')

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
                                           year,
                                           doi.replace('/', '@') + '.md')
                    os.makedirs(os.path.dirname(md_path), exist_ok=True)
                    save_to_md(md_dict, md_path)
                    with open(md_path, 'r', encoding='utf-8') as f:
                        md_text = f.read()

                    st.session_state['md_text'] = md_text
                    st.session_state['pdf_md_submit_disable'] = False

                st.success('PDF识别完毕')

    st.markdown(' ')
    with st.container(border=True):
        md_col1, md_col2 = st.columns([1, 1], gap='medium')

        md_value = md_col1.text_area('文本内容',
                                     height=800,
                                     key='md_text',
                                     disabled=st.session_state['pdf_uploader_disable'],
                                     label_visibility='collapsed')

        if md_value is not None:
            md_col2.container(height=800).write(md_value)

        submit = st.button('添加文献', type='primary', disabled=st.session_state['pdf_md_submit_disable'])

        if submit:
            if st.session_state['md_text']:
                with st.spinner('Adding markdown to vector db...'):
                    doc = split_markdown_text(md_value,
                                              year=st.session_state['paper_info']['year'],
                                              doi=st.session_state['paper_info']['doi'],
                                              author='')  # todo
                    config.set_collection(option)
                    st.cache_resource.clear()
                    vector_db = load_vectorstore()
                    vector_db.add_documents(doc)

                st.success('添加完成')
            else:
                st.warning('输入不能为空')


def pmc_tab():
    col_1, col_2, col_3 = st.columns([1.2, 3, 0.8], gap='medium')

    with col_1.container(border=True):
        st.subheader('使用说明')
        st.markdown(
            """   
            考虑到网络因素，目前暂时仅支持下载单个文献
            1. 输入PMC编号，如`PMC5386761`
            2. 等待解析完成即可
            """
        )

    with col_2.container(border=True):
        st.markdown('选择知识库')
        option = st.selectbox('选择知识库',
                              range(len(collections)),
                              format_func=lambda x: collections[x],
                              disabled=st.session_state['pmc_uploader_disable'],
                              label_visibility='collapsed')

        st.markdown('PMC ID')
        pmc_col1, pmc_col2 = st.columns([3, 1], gap='large')
        pmc_id = pmc_col1.text_input('PMC ID',
                                     key='pmc_id',
                                     disabled=st.session_state['pmc_uploader_disable'],
                                     label_visibility='collapsed')
        pmc_col2.checkbox('构建引用树', key='build_ref_tree', disabled=st.session_state['pmc_uploader_disable'])

        st.button('下载并添加', type='primary', key='pmc_submit', disabled=st.session_state['pmc_uploader_disable'])

        if st.session_state.get('pmc_submit'):
            config.set_collection(option)
            with st.spinner('Downloading paper...'):
                dl = download_paper_data(pmc_id)

            doi = dl['doi']
            year = dl['year']
            xml_path = dl['output_path']

            with st.spinner('Parsing paper...'):
                with open(xml_path, 'r', encoding='utf-8') as f:
                    xml_text = f.read()

                data = parse_paper_data(xml_text, year, doi)

                output_path = os.path.join(config.get_md_path(), year, f"{doi.replace('/', '@')}.md")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_to_md(data['sections'], output_path, ref=True, year=year, author=data['author'], doi=doi)

            with st.spinner('Adding paper to vector db...'):
                doc = section_to_documents(data['sections'], year=int(year), doi=doi, author=data['author'])
                st.cache_resource.clear()

                vector_db = load_vectorstore()
                doc_db = load_doc_store()
                retriever = base_retriever(vector_db, doc_db)

                retriever.add_documents(doc)

            if st.session_state.get('build_ref_tree'):
                # TODO
                pass

            st.success('添加完成')


tab1, tab2, tab3, tab4 = st.tabs(['Markdown', 'PDF', 'Pubmed Center', 'arXiv'])

with tab1:
    markdown_tab()

with tab2:
    pdf_tab()

with tab3:
    pmc_tab()
