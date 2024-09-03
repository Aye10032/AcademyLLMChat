from datetime import datetime

import pandas as pd
import streamlit as st
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from Config import Collection, UserRole, Config
from llm.ModelCore import load_embedding
from storage.MilvusConnection import MilvusConnection
from storage.SqliteStore import SqliteDocStore
from uicomponent.StComponent import side_bar_links, role_check
from uicomponent.StatusBus import get_config, update_config
from utils.FileUtil import is_en
from storage.MilvusParams import IndexType, get_index_param

st.set_page_config(
    page_title='学术大模型知识库',
    page_icon='📖',
    layout='centered',
    menu_items={
        'Report a bug': 'https://github.com/Aye10032/AcademyLLMChat/issues',
        'About': 'https://github.com/Aye10032/AcademyLLMChat'
    }
)

config: Config = get_config()
milvus_cfg = config.milvus_config
collections = []
for collection in milvus_cfg.collections:
    collections.append(collection.collection_name)

dtype = {
    0: 'NONE',
    1: 'BOOL',
    2: 'INT8',
    3: 'INT16',
    4: 'INT32',
    5: 'INT64',
    10: 'FLOAT',
    11: 'DOUBLE',
    20: 'STRING',
    21: 'VARCHAR',
    22: 'ARRAY',
    23: 'JSON',
    100: 'BINARY_VECTOR',
    101: 'FLOAT_VECTOR',
    999: 'UNKNOWN'
}

with st.sidebar:
    side_bar_links()

role_check(UserRole.OWNER)


def del_collection(option: int) -> None:
    target_collection = milvus_cfg.collections[option]
    with MilvusConnection(**milvus_cfg.get_conn_args()) as conn:
        conn.drop_collection(target_collection.collection_name)

    milvus_cfg.remove_collection(option)
    update_config(config)
    st.session_state['verify_text'] = ''


def rename_collection(option: int):
    milvus_cfg.rename_collection(option, st.session_state['col_title'])
    update_config(config)


def change_visible(option: int) -> None:
    milvus_cfg.set_collection_visibility(option, st.session_state.col_visible)
    update_config(config)


def manage_tab():
    st.header('知识库信息')
    option = st.selectbox('选择知识库',
                          range(len(collections)),
                          format_func=lambda x: collections[x])

    if option is not None:
        with MilvusConnection(**milvus_cfg.get_conn_args()) as conn:
            collection_name = milvus_cfg.collections[option].collection_name
            st.write(f'知识库 {collection_name} 中共有', conn.get_entity_num(collection_name), '条向量数据')
            field_df = pd.DataFrame()
            for index, field in enumerate(conn.get_collection(collection_name).schema.fields):
                df = pd.DataFrame(
                    {
                        '字段': field.name,
                        '类型': dtype[field.dtype],
                        'max_length': field.max_length,
                        'dim': field.dim,
                        'is_primary': field.is_primary,
                        'auto_id': field.auto_id,
                    },
                    index=[index]
                )

                field_df = pd.concat([field_df, df], ignore_index=True)

        st.dataframe(field_df, hide_index=True)

        st.markdown(' ')

        st.markdown('**数据库查询界面标题**')
        renam_col1, renam_col2 = st.columns([3, 1], gap='large')
        renam_col1.text_input(
            '数据库查询界面标题',
            milvus_cfg.collections[option].title,
            key='col_title',
            disabled=st.session_state['manage_collection_disable'],
            label_visibility='collapsed'
        )
        renam_col2.button(
            'Rename',
            on_click=rename_collection,
            args=[option],
            disabled=st.session_state['manage_collection_disable']
        )

        st.markdown(' ')

        st.markdown('**变更数据库用户可见性**')
        st.toggle(
            '可见性',
            milvus_cfg.collections[option].visitor_visible,
            key='col_visible',
            on_change=change_visible,
            args=[option],
            disabled=st.session_state['manage_collection_disable'],
            label_visibility='collapsed'
        )

        if st.session_state.col_visible:
            st.caption('当前数据库对普通用户可见')
        else:
            st.caption('当前数据库对普通用户不可见')

        st.markdown(' ')

        st.subheader(':red[危险操作]')

        with st.container(border=True):
            st.markdown('**删除知识库**')
            drop_verify = st.text_input('collection name',
                                        disabled=st.session_state['manage_collection_disable'],
                                        label_visibility='collapsed',
                                        key='verify_text')
            st.caption(f'若确定要删除知识库，请在此输入 `{collection_name}`')

            if drop_verify == collection_name:
                st.session_state['drop_collection_disable'] = False
            else:
                st.session_state['drop_collection_disable'] = True

            st.button(
                '删除知识库',
                type='primary',
                disabled=st.session_state['drop_collection_disable'],
                on_click=del_collection,
                kwargs={
                    'option': option
                }
            )


def new_tab():
    st.header('新建知识库')

    with st.container(border=True):
        col1_1, col1_2 = st.columns([3, 1], gap='medium')
        collection_name = col1_1.text_input('知识库名称 :red[*]', disabled=st.session_state['new_collection_disable'])
        language = col1_2.selectbox('语言', ['en', 'zh'], disabled=st.session_state['new_collection_disable'])

        col2_1, col2_2 = st.columns([3, 1], gap='medium')
        title = col2_1.text_input('页面名称', disabled=st.session_state['new_collection_disable'])
        visible = col2_2.toggle('用户可见', True)
        description = st.text_area('collection 描述', disabled=st.session_state['new_collection_disable'])

        with st.expander('向量库参数设置'):
            col2_1, col2_2 = st.columns(2, gap='medium')
            metric_type = col2_1.selectbox('Metric Type',
                                           ['L2', 'IP'],
                                           disabled=st.session_state['new_collection_disable'])

            index_types = ['IVF_FLAT',
                           'IVF_SQ8',
                           'IVF_PQ',
                           'HNSW',
                           'RHNSW_FLAT',
                           'RHNSW_SQ',
                           'RHNSW_PQ',
                           'IVF_HNSW',
                           'ANNOY',
                           'AUTOINDEX']
            index_type = col2_2.selectbox('Index Type',
                                          range(len(index_types)),
                                          format_func=lambda x: index_types[x],
                                          index=IndexType.HNSW,
                                          disabled=st.session_state['new_collection_disable'])

            if index_type is not None:
                param = st.text_area('params', value=get_index_param(index_type),
                                     disabled=st.session_state['new_collection_disable'])

        if st.button(
                '新建知识库',
                type='primary',
                disabled=st.session_state['new_collection_disable'],
        ):
            if not (collection_name and is_en(collection_name)):
                st.error('知识库名称必须是不为空的英文')
                st.stop()

            with MilvusConnection(**milvus_cfg.get_conn_args()) as conn:
                if conn.has_collection(collection_name):
                    st.error('知识库已存在')
                    st.stop()

            if not title:
                title = collection_name

            if not description:
                description = f'This is a collection about {collection_name}'

            index_param = {
                "metric_type": metric_type,
                "index_type": index_types[index_type],
                "params": eval(param),
            }

            embedding = load_embedding()

            with st.spinner('Creating collection...'):
                new_collection = Collection.from_dict({
                    "collection_name": collection_name,
                    "language": language,
                    "title": title,
                    "description": description,
                    "index_param": index_param,
                    "visitor_visible": visible,
                })

                init_doc = Document(page_content=f'This is a collection about {description}',
                                    metadata={
                                        'title': 'About this collection',
                                        'section': 'Abstract',
                                        'author': 'administrator',
                                        'year': datetime.now().year,
                                        'type': -1,
                                        'keywords': 'collection',
                                        'doi': ''
                                    })

                sqlite_path = config.get_sqlite_path(new_collection.collection_name)
                doc_store = SqliteDocStore(
                    connection_string=sqlite_path,
                    drop_old=True
                )

                vector_db = Milvus(
                    embedding,
                    collection_name=collection_name,
                    connection_args=milvus_cfg.get_conn_args(),
                    index_params=index_param,
                    drop_old=True,
                    auto_id=True
                )

                if language == 'en':
                    child_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=100,
                        chunk_overlap=0,
                        separators=['.', '\n\n', '\n'],
                        keep_separator=False
                    )
                else:
                    child_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=100,
                        chunk_overlap=0,
                        separators=['。', '？', '\n\n', '\n'],
                        keep_separator=False
                    )

                retriever = ParentDocumentRetriever(
                    vectorstore=vector_db,
                    docstore=doc_store,
                    child_splitter=child_splitter
                )

                retriever.add_documents([init_doc])

                milvus_cfg.add_collection(new_collection)
                config.set_collection(0)
                update_config(config)
            logger.info('success')
            st.success('创建成功')
            st.balloons()
            st.rerun()


def user_tab():
    st.title('用户管理')


tab1, tab2, tab3 = st.tabs(['知识库管理', '新建知识库', '用户管理'])

with tab1:
    manage_tab()

with tab2:
    new_tab()

with tab3:
    user_tab()
