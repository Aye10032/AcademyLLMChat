from datetime import datetime

import pandas as pd
import streamlit as st
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.documents import Document
from loguru import logger

from Config import config, Collection
from llm.ModelCore import load_embedding_zh, load_embedding_en
from vectorstore.MilvusConnection import MilvusConnection
from vectorstore.MilvusParams import IndexType, get_index_param

st.set_page_config(
    page_title='学术大模型知识库',
    page_icon='📖',
    layout='centered'
)

milvus_cfg = config.milvus_config
collections = []
for collection in milvus_cfg.COLLECTIONS:
    collections.append(collection.NAME)

if milvus_cfg.USING_REMOTE:
    uri = milvus_cfg.REMOTE_DATABASE['url']
    user = milvus_cfg.REMOTE_DATABASE['username']
    password = milvus_cfg.REMOTE_DATABASE['password']
    conn = st.connection('milvus', type=MilvusConnection,
                         uri=uri,
                         user=user,
                         password=password,
                         secure=True)
    connection_args = {
        'uri': milvus_cfg.REMOTE_DATABASE['url'],
        'user': milvus_cfg.REMOTE_DATABASE['username'],
        'password': milvus_cfg.REMOTE_DATABASE['password'],
        'secure': True,
    }
else:
    conn = st.connection('milvus', type=MilvusConnection,
                         uri=f'http://{milvus_cfg.MILVUS_HOST}:{milvus_cfg.MILVUS_PORT}')
    connection_args = {
        'host': milvus_cfg.MILVUS_HOST,
        'port': milvus_cfg.MILVUS_PORT,
    }

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
    st.header('欢迎使用学术LLM知识库')

    st.page_link('App.py', label='Home', icon='💬')
    st.page_link('pages/FileUpload.py', label='上传文件', icon='📂')
    st.page_link('pages/CollectionManage.py', label='知识库管理', icon='🖥️')

    st.divider()


def manage_tab():
    st.header('知识库管理')
    option = st.selectbox('选择知识库',
                          range(len(collections)),
                          format_func=lambda x: collections[x])

    if option is not None:
        collection_name = milvus_cfg.COLLECTIONS[option].NAME
        st.write(f'知识库 {collection_name} 中共有', conn.get_entity_num(collection_name), '条向量数据')
        field_df = pd.DataFrame()
        for index, field in enumerate(conn.get_collection(collection_name).schema.fields):
            df = pd.DataFrame(
                {
                    'name': field.name,
                    'type': dtype[field.dtype],
                    'description': field.description,
                    'max_length': field.max_length,
                    'dim': field.dim,
                    'is_primary': field.is_primary,
                    'auto_id': field.auto_id,
                },
                index=[index]
            )

            field_df = pd.concat([field_df, df], ignore_index=True)

        st.dataframe(field_df, hide_index=True)


def new_tab():
    st.header('新建知识库')
    with st.container(border=True):
        col1_1, col1_2 = st.columns([3, 1], gap='medium')
        collection_name = col1_1.text_input('知识库名称 :red[*]')
        language = col1_2.selectbox('语言', ['en', 'zh'])

        title = st.text_input('页面名称')
        description = st.text_area('collection 描述')

        with st.expander('向量库参数设置'):
            col2_1, col2_2 = st.columns(2, gap='medium')
            metric_type = col2_1.selectbox('Metric Type', ['L2', 'IP'])

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
                                          index=IndexType.HNSW)

            if index_type is not None:
                # print(index_type)
                param = st.text_area('params', value=get_index_param(index_type))

        if submit := st.button('新建知识库', type='primary'):
            # conn.create_collection(collection_name)
            if not (collection_name and collection_name.isalpha()):
                st.error('知识库名称必须是不为空的英文')
                st.stop()

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

            if language == 'zh':
                embedding = load_embedding_zh()
            else:
                embedding = load_embedding_en()

            with st.spinner('Creating collection...'):
                doc = Document(page_content=description,
                               metadata={'Title': 'About this collection', 'Section': 'Abstract', 'doi': 'empty',
                                         'year': datetime.now().year})
                vector_db = Milvus.from_documents(
                    [doc],
                    embedding,
                    collection_name=collection_name,
                    connection_args=connection_args,
                    drop_old=True
                )

                milvus_cfg.add_collection(
                    Collection.from_dict({"collection_name": collection_name,
                                          "language": language,
                                          "title": title,
                                          "description": description,
                                          "index_param": index_param}))
            logger.info('success')
            st.success('创建成功')
            st.balloons()


tab1, tab2 = st.tabs(['知识库管理', '新建知识库'])

with tab1:
    manage_tab()

with tab2:
    new_tab()
