import streamlit as st
from st_pages import show_pages_from_config

st.set_page_config(page_title="微藻文献大模型知识库", page_icon="📖", layout='centered')
st.title('添加文献')

with st.sidebar:
    show_pages_from_config()
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

uploaded_files = st.file_uploader('选择PDF或markdown文件', type=['md', 'pdf'], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # string_data = stringio.read()
    # st.markdown(string_data)
