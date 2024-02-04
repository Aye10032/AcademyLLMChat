import streamlit as st

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
