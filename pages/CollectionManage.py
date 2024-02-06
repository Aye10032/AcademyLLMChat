import os

import pandas as pd
import streamlit as st

from Config import config
from utils.MilvusConnection import MilvusConnection

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
else:
    conn = st.connection('milvus', type=MilvusConnection,
                         uri=f'http://{milvus_cfg.MILVUS_HOST}:{milvus_cfg.MILVUS_PORT}')

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
    st.warning('此页面新建的只是索引，需要在上传界面至少添加一个文件后才会在向量库中实际建立collection并进行查询')


tab1, tab2 = st.tabs(['知识库管理', '新建知识库'])

with tab1:
    manage_tab()

with tab2:
    new_tab()
