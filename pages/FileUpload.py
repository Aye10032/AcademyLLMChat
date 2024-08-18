import os
import shutil
from datetime import datetime
from typing import Tuple, List

import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from loguru import logger
from pandas import DataFrame
from tqdm import tqdm

from Config import UserRole, Config, MilvusConfig, Collection
from llm.ModelCore import load_embedding
from llm.RagCore import load_vectorstore, load_doc_store
from llm.RetrieverCore import insert_retriever
from llm.storage.MilvusConnection import MilvusConnection
from llm.storage.SqliteStore import ReferenceStore
from uicomponent.StComponent import side_bar_links, role_check
from uicomponent.StatusBus import get_config
from utils.paper.Paper import *

import utils.MarkdownPraser as md
import utils.GrobidUtil as gb
import utils.PubmedUtil as pm
import utils.PMCUtil as pmc

config: Config = get_config()
milvus_cfg: MilvusConfig = config.milvus_config
collections = []
for collection in milvus_cfg.collections:
    collections.append(collection.collection_name)

st.set_page_config(
    page_title="学术大模型知识库",
    page_icon="📖",
    layout='wide',
    menu_items={
        'Report a bug': 'https://github.com/Aye10032/AcademyLLMChat/issues',
        'About': 'https://github.com/Aye10032/AcademyLLMChat'
    }
)

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


def __download_reference(target_collection: Collection, ref_list: DataFrame):
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
            tag, pmid = __download_from_pmc(target_collection, row.pmc)
            if tag == 0:
                ref_list.at[index, 'exist'] = True
            else:
                ref_list.at[index, 'exist'] = False
            ref_list.at[index, 'download'] = True
            st.session_state['ref_list'] = ref_list

        progress_num = (int(str(index)) + 1) / file_count
        ref_bar.progress(progress_num, text=f'正在处理文本({int(str(index)) + 1}/{file_count})，请勿关闭或刷新此页面')

    ref_bar.empty()


def __download_from_pmc(target_collection: Collection, pmc_id: str, is_reference: bool = True) -> Tuple[int, Reference]:
    with st.spinner('Downloading paper...'):
        _, dl = pmc.download_paper_data(pmc_id, config)

    doi = dl['doi']
    year = dl['year']
    xml_path = dl['output_path']

    if xml_path is None:
        return -1, Reference(doi, [])

    with st.spinner('Parsing paper...'):
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_text = f.read()

        flag, data = pmc.parse_paper_data(xml_text)

        if not flag:
            return -1, Reference(doi, [])

        output_path = os.path.join(config.get_md_path(target_collection.collection_name), year, f"{doi.replace('/', '@')}.md")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not is_reference:
        data.info.ref = True

        with st.spinner('Analysing reference...'):
            ref_list = data.reference.ref_list
            for index, ref_dict in enumerate(ref_list):
                ref_dict: dict
                if term := ref_dict.get('pmid') != '':
                    ref_dict = pm.get_info_by_term(term, pm.SearchType.PM)
                elif term := ref_dict.get('doi') != '':
                    ref_dict = pm.get_info_by_term(term, pm.SearchType.DOI)
                elif term := ref_dict.get('title') != '':
                    ref_dict = pm.get_info_by_term(term, pm.SearchType.TITLE)
                else:
                    continue

                ref_list[index] = ref_dict

            data.reference.ref_list = ref_list
    else:
        data.info.ref = False

    md.save_to_md(data, output_path)

    with st.spinner('Adding paper to vector db...'):
        docs, ref_data = md.split_paper(data)
        if not is_reference:
            __add_documents(target_collection, docs, ref_data)
        else:
            __add_documents(target_collection, docs)

    return 0, ref_data


def __add_documents(target_collection: Collection, docs: list[Document], ref_data: Reference = None) -> None:
    embedding = load_embedding()
    vector_db = load_vectorstore(target_collection.collection_name, embedding)
    doc_db = load_doc_store(config.get_sqlite_path(target_collection.collection_name))
    retriever = insert_retriever(vector_db, doc_db, target_collection.language)
    retriever.add_documents(docs)

    if (
            st.session_state.get('build_ref_tree')
            and
            ref_data is not None
            and
            len(ref_data.ref_list) > 0
    ):
        with ReferenceStore(config.get_reference_path()) as ref_store:
            ref_store.add_reference(ref_data)


def set_ref_build(key_word: str):
    if 'build_ref_tree' not in st.session_state:
        st.session_state['build_ref_tree'] = False

    st.session_state['build_ref_tree'] = st.session_state.get(key_word)


def markdown_tab():
    col_1, col_2, col_3 = st.columns([1.2, 3, 0.5], gap='medium')

    with col_1.container(border=True):
        st.subheader('使用说明')
        st.markdown(
            """   
            先在本地手动将文献转为markdown文件之后再导入知识库，可以批量导入
            1. 在markdown文件的Front Matter中编辑相关信息，具体包括：
                - year: 文献发表年份      
                - doi: 文献DOI号       
                - author: 文献作者(目前仅支持单一作者)        
            2. 上传markdown文件      
            3. 点击导入文献按钮，等待处理完成     
            """
        )

    with (col_2.container(border=True)):
        st.markdown('选择知识库')

        md_col1, md_col2 = st.columns([3, 1], gap='large')
        md_col1.selectbox(
            '选择知识库',
            range(len(collections)),
            format_func=lambda x: collections[x],
            key='md_selection',
            disabled=st.session_state['md_uploader_disable'],
            label_visibility='collapsed'
        )

        md_col2.checkbox(
            '构建引用树',
            key='md_build_ref_tree',
            disabled=st.session_state['pdf_uploader_disable'],
            on_change=set_ref_build,
            kwargs={'key_word': 'md_build_ref_tree'}
        )

        st.markdown(' ')

        st.markdown('上传markdown文件')
        uploaded_files = st.file_uploader(
            '上传Markdown文件',
            type=['md'],
            accept_multiple_files=True,
            disabled=st.session_state['md_uploader_disable'],
            label_visibility='collapsed'
        )

        st.button('导入文献', key='md_submit', type='primary', disabled=st.session_state['md_uploader_disable'])

        if st.session_state.get('md_submit'):
            option = st.session_state.get('md_selection')
            target_collection = milvus_cfg.get_collection_by_id(option)

            file_count = len(uploaded_files)
            if file_count == 0:
                st.warning('还没有上传文件')
                st.stop()

            progress_text = f'正在处理文献(0/{file_count})，请勿关闭或刷新此页面'
            md_bar = st.progress(0, text=progress_text)
            for index, uploaded_file in tqdm(enumerate(uploaded_files), total=file_count):
                doc, ref_data = md.split_markdown(uploaded_file)

                doi = doc[0].metadata.get('doi')
                year = doc[0].metadata.get('year')
                if not doi == '':
                    with MilvusConnection(**milvus_cfg.get_conn_args()) as conn:
                        _num = len(conn.client.query(
                            target_collection.collection_name,
                            filter=f'doi == "{doi}"'
                        ))
                    if _num > 0:
                        st.error(f'文章 {doi} 已存在，跳过')
                        continue

                __add_documents(target_collection, doc, ref_data)

                file_path = os.path.join(config.get_md_path(target_collection.collection_name), str(year), uploaded_file.name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                progress_num = (index + 1) / file_count
                md_bar.progress(progress_num, text=f'正在处理文本({index + 1}/{file_count})，请勿关闭或刷新此页面')
            md_bar.empty()
            st.write('处理完成，共添加了', file_count, '份文献')


def pdf_tab():
    col_1, col_2, col_3 = st.columns([1.2, 3, 0.5], gap='medium')

    with col_1.container(border=True):
        st.subheader('使用说明')
        st.markdown(
            """        
            1. 确保grobid已经在运行     
            2. 上传PDF文件      
                - (可选)选择构建引用树
            3. 解析并添加文献
            4. 等待处理完成  
            """
        )

    with (col_2.container(border=True)):
        st.markdown('选择知识库')
        pdf_col1, pdf_col2 = st.columns([3, 1], gap='large')
        pdf_col1.selectbox(
            '选择知识库',
            range(len(collections)),
            format_func=lambda x: collections[x],
            key='pdf_selection',
            disabled=st.session_state['pdf_uploader_disable'],
            label_visibility='collapsed'
        )

        pdf_col2.checkbox(
            '构建引用树',
            key='pdf_build_ref_tree',
            disabled=st.session_state['pdf_uploader_disable'],
            on_change=set_ref_build,
            kwargs={'key_word': 'pdf_build_ref_tree'}
        )

        uploaded_files = st.file_uploader(
            '选择PDF文件',
            type=['pdf'],
            disabled=st.session_state['pdf_uploader_disable'],
            accept_multiple_files=True
        )

        st.button('解析并添加', key='pdf_submit', type='primary', disabled=st.session_state['pdf_uploader_disable'])

        if st.session_state.get('pdf_submit'):

            file_count = len(uploaded_files)
            if file_count == 0:
                st.warning('请先上传PDF文件')
                st.stop()

            option = st.session_state.get('pdf_selection')
            target_collection = milvus_cfg.get_collection_by_id(option)
            target_name = target_collection.collection_name

            os.makedirs(os.path.join(config.get_pdf_path(target_name), 'unknown'), exist_ok=True)
            os.makedirs(os.path.join(config.get_xml_path(target_name), 'unknown'), exist_ok=True)
            os.makedirs(os.path.join(config.get_md_path(target_name), 'unknown'), exist_ok=True)

            progress_text = f'正在处理文献(0/{file_count})，请勿关闭或刷新此页面'
            pdf_bar = st.progress(0, text=progress_text)
            for index, uploaded_file in tqdm(enumerate(uploaded_files), total=file_count):
                pdf_path = os.path.join(
                    config.get_pdf_path(target_name),
                    'unknown',
                    uploaded_file.name
                )
                with open(pdf_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                _, _, xml_text = gb.parse_pdf_to_xml(pdf_path, config)

                xml_path = os.path.join(
                    config.get_xml_path(target_name),
                    'unknown',
                    uploaded_file.name.replace('.pdf', '.xml')
                )
                with open(xml_path, 'w', encoding='utf-8') as f:
                    f.write(xml_text)

                result = gb.parse_xml(xml_path)

                year = result.info.year
                doi = result.info.doi

                if not doi == '':
                    with MilvusConnection(**milvus_cfg.get_conn_args()) as conn:
                        _num = len(conn.client.query(
                            target_collection.collection_name,
                            filter=f'doi == "{doi}"'
                        ))
                    if _num > 0:
                        st.error(f'文章 {doi} 已存在，跳过')
                        continue

                if year != -1:
                    md_path = os.path.join(
                        config.get_md_path(target_name),
                        str(year),
                        f"{doi.replace('/', '@')}.md"
                    )
                    os.makedirs(os.path.dirname(md_path), exist_ok=True)

                    new_pdf_path = os.path.join(
                        config.get_pdf_path(target_name),
                        str(year),
                        f"{doi.replace('/', '@')}.pdf"
                    )
                    os.makedirs(os.path.dirname(new_pdf_path), exist_ok=True)
                    shutil.move(pdf_path, new_pdf_path)

                    new_xml_path = os.path.join(
                        config.get_xml_path(target_name),
                        str(year),
                        f"{doi.replace('/', '@')}.xml"
                    )
                    os.makedirs(os.path.dirname(new_xml_path), exist_ok=True)
                    shutil.move(xml_path, new_xml_path)
                else:
                    md_path = os.path.join(
                        config.get_md_path(target_name),
                        'unknown',
                        f"{doi.replace('/', '@')}.md"
                    )

                md.save_to_md(result, md_path)

                # ----------------------------------
                docs, ref_data = md.split_paper(result)

                if st.session_state.get('pdf_build_ref_tree'):
                    with st.spinner('Analysing reference...'):
                        ref_list = ref_data.ref_list
                        for ref_idx, ref_dict in enumerate(ref_list):
                            ref_dict: dict
                            if term := ref_dict.get('pmid') != '':
                                ref_dict = pm.get_info_by_term(term, pm.SearchType.PM)
                            elif term := ref_dict.get('doi') != '':
                                ref_dict = pm.get_info_by_term(term, pm.SearchType.DOI)
                            elif term := ref_dict.get('title') != '':
                                ref_dict = pm.get_info_by_term(term, pm.SearchType.TITLE)
                            else:
                                continue

                            ref_list[ref_idx] = ref_dict

                        ref_data.ref_list = ref_list

                    with st.spinner('Adding document to database...'):
                        __add_documents(target_collection, docs, ref_data)

                    # TODO 引用文献下载
                    # ref_list = __check_exist(pd.DataFrame(ref_list))
                    #
                    # try:
                    #     __download_reference(ref_list)
                    # except Exception as e:
                    #     logger.error(e)
                    #     st.session_state['retry_disable'] = False
                    #     st.rerun()
                    # finally:
                    #     df_block.dataframe(st.session_state.get('ref_list'), use_container_width=True)
                    #     st.toast('引用处理完毕', icon='👏')
                else:
                    with st.spinner('Adding document to database...'):
                        __add_documents(target_collection, docs)

                progress_num = (index + 1) / file_count
                pdf_bar.progress(progress_num, text=f'正在处理文本({index + 1}/{file_count})，请勿关闭或刷新此页面')

            pdf_bar.empty()
            st.success('文献添加完毕')
            st.snow()


def pmc_tab():
    col_1, col_2, col_3 = st.columns([1.2, 3, 0.5], gap='medium')

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

    with (col_2.container(border=True)):
        st.markdown('选择知识库')
        pmc_col1, pmc_col2 = st.columns([3, 1], gap='large')
        pmc_col1.selectbox(
            '选择知识库',
            range(len(collections)),
            format_func=lambda x: collections[x],
            key='pmc_selection',
            disabled=st.session_state['pmc_uploader_disable'],
            label_visibility='collapsed'
        )

        pmc_col2.checkbox(
            '构建引用树',
            key='pmc_build_ref_tree',
            disabled=st.session_state['pmc_uploader_disable'],
            on_change=set_ref_build,
            kwargs={'key_word': 'pmc_build_ref_tree'}
        )

        st.markdown('PMC ID')
        pmc_id = st.text_input(
            'PMC ID',
            disabled=st.session_state['pmc_uploader_disable'],
            label_visibility='collapsed'
        )

        st.button('下载并添加', type='primary', key='pmc_submit', disabled=st.session_state['pmc_uploader_disable'])

        if st.session_state.get('pmc_submit'):
            option = st.session_state.get('pmc_selection')
            target_collection = milvus_cfg.get_collection_by_id(option)

            tag, pmid = __download_from_pmc(target_collection, pmc_id)

            if tag == -1:
                st.error('文章结构不完整！请检查相关信息，或尝试通过PDF加载.')

            if st.session_state.get('pmc_build_ref_tree'):
                with st.spinner('Analysing reference...'):
                    # TODO 引用文献下载
                    pass

            st.success('添加完成')


tab1, tab2, tab3, tab4 = st.tabs(['Markdown', 'PDF', 'Pubmed Center', 'arXiv'])

with tab1:
    markdown_tab()

with tab2:
    pdf_tab()

with tab3:
    pmc_tab()
