import os
from datetime import datetime

import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from loguru import logger
from pandas import DataFrame
from tqdm import tqdm

from Config import UserRole, Config, MilvusConfig
from llm.RagCore import load_vectorstore, load_doc_store
from llm.RetrieverCore import base_retriever
from llm.storage.MilvusConnection import MilvusConnection
from uicomponent.StComponent import side_bar_links, role_check
from uicomponent.StatusBus import update_config, get_config
from utils.FileUtil import save_to_md, section_to_documents, Section
from utils.GrobidUtil import parse_xml, parse_pdf_to_xml
from utils.MarkdownPraser import split_markdown, split_markdown_text
from utils.PMCUtil import download_paper_data, parse_paper_data
from utils.PubmedUtil import get_paper_info

config: Config = get_config()
milvus_cfg: MilvusConfig = config.milvus_config
collections = []
for collection in milvus_cfg.COLLECTIONS:
    collections.append(collection.NAME)

st.set_page_config(page_title="学术大模型知识库", page_icon="📖", layout='wide')

with st.sidebar:
    side_bar_links()

    st.subheader('构建引用树功能')
    st.markdown("""
    目前，仅有PDF、PMC、arXiv(待实现)支持引用树构建的功能。
    
    若选择构建引用树，则会自动下载引用文献中拥有PMC full free text的文献，并加入知识库。     
    
    若因网络原因出现下载失败，请不要刷新界面，点击重试按钮，再次尝试进行引用文献的下载。      
    """)

role_check(UserRole.ADMIN, True)

st.title('添加文献')

if 'ref_list' not in st.session_state:
    st.session_state['ref_list'] = None

if 'retry_disable' not in st.session_state:
    st.session_state['retry_disable'] = True


def __check_exist(ref_list: DataFrame) -> DataFrame:
    ref_list['exist'] = False
    with MilvusConnection(**milvus_cfg.get_conn_args()) as conn:
        for index, row in ref_list.iterrows():
            _num = conn.query(milvus_cfg.get_collection().NAME, filter=f'doi == "{row.doi}"')
            if len(_num) != 0:
                ref_list.at[index, 'exist'] = True

    return ref_list


def __download_reference(ref_list: DataFrame):
    st.session_state['retry_disable'] = True

    ref_bar = st.progress(0, text='')
    file_count = ref_list.shape[0]

    for index, row in ref_list.iterrows():

        if row.download or row.exist:
            logger.info(f'{row.doi} exist, skip.')
            ref_list.at[index, 'download'] = True
            st.session_state['ref_list'] = ref_list
            continue

        if pd.notna(row.pmc):
            tag = __download_from_pmc(row.pmc)
            if tag == 0:
                ref_list.at[index, 'exist'] = True
            else:
                ref_list.at[index, 'exist'] = False
            ref_list.at[index, 'download'] = True
            st.session_state['ref_list'] = ref_list

        progress_num = (int(str(index)) + 1) / file_count
        ref_bar.progress(progress_num, text=f'正在处理文本({int(str(index)) + 1}/{file_count})，请勿关闭或刷新此页面')

    ref_bar.empty()


def __download_from_pmc(pmc_id: str) -> int:
    with st.spinner('Downloading paper...'):
        _, dl = download_paper_data(pmc_id, config)

    doi = dl['doi']
    year = dl['year']
    xml_path = dl['output_path']

    if xml_path is None:
        return -1

    with st.spinner('Parsing paper...'):
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_text = f.read()

        data = parse_paper_data(xml_text, year, doi)

        if not data['norm']:
            return -1

        output_path = os.path.join(config.get_md_path(), year, f"{doi.replace('/', '@')}.md")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_to_md(data['sections'], output_path, ref=True, year=year, author=data['author'], doi=doi)

    with st.spinner('Adding paper to vector db...'):
        docs = section_to_documents(data['sections'], year=int(year), doi=doi, author=data['author'])
        __add_documents(docs)

    return 0


def __add_documents(docs: list[Document]) -> None:
    vector_db = load_vectorstore(config.milvus_config.get_collection().NAME)
    doc_db = load_doc_store()
    retriever = base_retriever(vector_db, doc_db)

    retriever.add_documents(docs)


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
            config.set_collection(option)
            update_config(config)

            file_count = len(uploaded_files)
            if file_count == 0:
                st.warning('还没有上传文件')
                st.stop()

            progress_text = f'正在处理文献(0/{file_count})，请勿关闭或刷新此页面'
            md_bar = st.progress(0, text=progress_text)
            for index, uploaded_file in tqdm(enumerate(uploaded_files), total=file_count):
                doc = split_markdown(uploaded_file)
                __add_documents(doc)
                progress_num = (index + 1) / file_count
                md_bar.progress(progress_num, text=f'正在处理文本({index + 1}/{file_count})，请勿关闭或刷新此页面')
            md_bar.empty()
            st.write('处理完成，共添加了', file_count, '份文献')


def pdf_tab():
    col_1, col_2, col_3 = st.columns([1.2, 3, 0.5], gap='medium')

    with col_1.container(border=True):
        st.subheader('使用说明')
        st.warning('由于PDF解析需要请求PubMed信息，为了防止大量访问造成解析失败，仅允许上传单个文件')
        st.markdown(
            """  
            1. 将PDF文件重命名为`PMxxxx.pdf`的格式          
            2. 确保grobid已经在运行     
            3. 上传PDF文件      
                - (可选)选择构建引用树
            4. 解析并添加文献
            5. 等待处理完成  
            """
        )

    with col_2.container(border=True):
        st.markdown('选择知识库')
        pdf_col1, pdf_col2 = st.columns([3, 1], gap='large')
        pdf_col1.selectbox('选择知识库',
                           range(len(collections)),
                           format_func=lambda x: collections[x],
                           key='pdf_selection',
                           disabled=st.session_state['pdf_uploader_disable'],
                           label_visibility='collapsed')
        pdf_col2.checkbox('构建引用树', key='pdf_build_ref_tree', disabled=st.session_state['pdf_uploader_disable'])

        uploaded_file = st.file_uploader('选择PDF文件',
                                         type=['pdf'],
                                         disabled=st.session_state['pdf_uploader_disable'])

        st.button('解析并添加', key='pdf_submit', type='primary', disabled=st.session_state['pdf_uploader_disable'])

        df_block = st.empty()
        retry_block = st.empty()
        error_block = st.empty()

        if st.session_state.get('ref_list') is not None:
            df_block.dataframe(st.session_state.get('ref_list'), use_container_width=True)

        if not st.session_state.get('retry_disable'):
            retry_block.button('重试', key='pdf_retry', disabled=st.session_state['retry_disable'])
            error_block.error('下载出现错误，请重试')

        if st.session_state.get('pdf_submit'):
            if uploaded_file is not None:
                option = st.session_state.get('pdf_selection')
                config.set_collection(option)
                update_config(config)

                with st.spinner('Download information from pubmed...'):
                    data = get_paper_info(uploaded_file.name.replace('.pdf', '').replace('PM', ''), config)

                    pdf_path = os.path.join(config.get_pdf_path(), data['year'], uploaded_file.name)
                    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

                    section_dict = [Section(data['title'], 1), Section('Abstract', 2), Section(data['abstract'], 0)]

                    with open(pdf_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    doi = data['doi']
                    year = data['year']
                    author = data['author']

                    xml_path = os.path.join(config.get_xml_path(), year, doi.replace('/', '@') + '.grobid.tei.xml')

                    os.makedirs(os.path.dirname(xml_path), exist_ok=True)

                with MilvusConnection(**milvus_cfg.get_conn_args()) as conn:
                    _num = conn.query(milvus_cfg.get_collection().NAME, filter=f'doi == "{doi}"')

                if len(_num) == 0:
                    with st.spinner('Parsing pdf...'):
                        _, _, xml_text = parse_pdf_to_xml(pdf_path, config)
                        with open(xml_path, 'w', encoding='utf-8') as f:
                            f.write(xml_text)
                        result = parse_xml(xml_path)

                        section_dict.extend(result['sections'])

                        md_path = os.path.join(config.get_md_path(), year, doi.replace('/', '@') + '.md')
                        os.makedirs(os.path.dirname(md_path), exist_ok=True)
                        save_to_md(section_dict, md_path, year=year, doi=doi, author=author, ref=False)

                        st.toast('PDF识别完毕', icon='👏')

                    docs = section_to_documents(section_dict, author, int(year), doi)
                    __add_documents(docs)
                    st.toast('PDF识别完毕', icon='👏')
                else:
                    st.info('向量库中已经存在此文献')

                if st.session_state.get('pdf_build_ref_tree'):
                    with st.spinner('Analysing reference...'):
                        ref_list = data['ref_list']
                        ref_list['download'] = False
                        ref_list = __check_exist(ref_list)

                        st.session_state['retry_visible'] = True
                        try:
                            __download_reference(ref_list)
                        except Exception as e:
                            logger.error(e)
                            st.session_state['retry_disable'] = False
                            st.rerun()
                        finally:
                            df_block.dataframe(st.session_state.get('ref_list'), use_container_width=True)
                            st.toast('引用处理完毕', icon='👏')

                st.success('文献添加完毕')
                st.snow()

            else:
                st.warning('请先上传PDF文件')

        if st.session_state.get('pdf_retry'):
            ref_list = st.session_state.get('ref_list')
            error_block.empty()

            with st.spinner('Analysing reference...'):
                try:
                    __download_reference(ref_list)
                except Exception as e:
                    logger.error(e)
                    st.session_state['retry_disable'] = False
                    st.rerun()
                finally:
                    st.success('引用处理完毕')


def pmc_tab():
    col_1, col_2, col_3 = st.columns([1.2, 3, 0.8], gap='medium')

    with col_1.container(border=True):
        st.subheader('使用说明')
        st.warning('考虑到网络稳定性因素，目前暂时仅支持下载单个文献')
        st.markdown(
            """   
            1. 输入PMC编号，如`PMC5386761`
                - (可选)选择构建引用树
            2. 下载并添加文献
            3. 等待解析完成即可
            """
        )

    with col_2.container(border=True):
        st.markdown('选择知识库')
        pmc_col1, pmc_col2 = st.columns([3, 1], gap='large')
        pmc_col1.selectbox('选择知识库',
                           range(len(collections)),
                           format_func=lambda x: collections[x],
                           key='pmc_selection',
                           disabled=st.session_state['pmc_uploader_disable'],
                           label_visibility='collapsed')

        pmc_col2.checkbox('构建引用树', key='build_ref_tree', disabled=st.session_state['pmc_uploader_disable'])

        st.markdown('PMC ID')
        pmc_id = st.text_input('PMC ID',
                               key='pmc_id',
                               disabled=st.session_state['pmc_uploader_disable'],
                               label_visibility='collapsed')

        st.button('下载并添加', type='primary', key='pmc_submit', disabled=st.session_state['pmc_uploader_disable'])

        if st.session_state.get('pmc_submit'):
            option = st.session_state.get('pmc_selection')
            config.set_collection(option)
            update_config(config)

            tag = __download_from_pmc(pmc_id)

            if tag == -1:
                st.error('文章结构不完整！请检查相关信息，或尝试通过PDF加载.')

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
