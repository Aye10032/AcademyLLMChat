import os

import pandas as pd
import streamlit as st
from st_milvus_connection import MilvusConnection

from Config import config

milvus_cfg = config.milvus_config
os.environ['milvus_uri'] = f'http://{milvus_cfg.MILVUS_HOST}:{milvus_cfg.MILVUS_PORT}'
os.environ['milvus_token'] = ''

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

st.title('知识库管理')

conn = st.connection("milvus", type=MilvusConnection)
df = (pd.DataFrame(conn.get_collection('Nannochloropsis').query(
    expr='year == 2012',
    output_fields=['Title', 'year', 'doi']
)).copy()
      .drop('pk', axis=1)
      .drop_duplicates(ignore_index=True))

st.dataframe(
    df,
    hide_index=True,
    column_order=['Title', 'year', 'doi']
)
