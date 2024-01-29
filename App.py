import streamlit as st

from llm.AgentCore import translate_sentence
from llm.RagCore import ask_from_rag
from llm.Template import TRANSLATE_TO_ZH

st.set_page_config(page_title='微藻文献大模型知识库', page_icon='📖', layout='centered')
st.title('微藻文献大模型知识库')

with st.sidebar:
    st.header('欢迎使用微藻文献知识库！')

prompt = st.chat_input('请输入问题')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    response= ask_from_rag(prompt)
    with st.chat_message('assistant'):
        st.markdown(response['result'])
        with st.expander('参考文献'):
            for ref in response['source_documents']:
                st.divider()
                _title = ref.metadata['Title']
                _year = ref.metadata['year']
                _doi = ref.metadata['doi']
                st.markdown(f'#### {_title}')
                st.caption(f'{_doi} ({_year})')
                st.markdown(ref.page_content)

    st.session_state.messages.append({'role': 'assistant', 'content': response['result']})
