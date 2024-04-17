import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from Config import Config
from uicomponent.StComponent import side_bar_links
from uicomponent.StatusBus import get_config
from llm.ModelCore import *

st.set_page_config(
    page_title='学术大模型知识库',
    page_icon='📖',
    layout='wide',
    menu_items={
        'Report a bug': 'https://github.com/Aye10032/AcademyLLMChat/issues',
        'About': 'https://github.com/Aye10032/AcademyLLMChat'
    }
)

config: Config = get_config()

st.title("一键生成翻译总结")

with st.sidebar:
    side_bar_links()

    st.selectbox('选择LLM',
                 options=['gpt3.5', 'gpt4', ''],
                 index=1,
                 key='TranslateLLM')


@st.cache_data(show_spinner="LLM answering...")
def get_translate_and_conclude(question: str, llm_name: str, step: int):
    if step == 0:
        _prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage("You are an AI academic assistant and should answer user questions rigorously."),
                ("human",
                 "首先，将这段文本**翻译为中文**，不要漏句。对于所有的特殊符号和latex代码，请保持原样不要改变:\n{question}"),
            ]
        )
    elif step == 1:
        _prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are an AI academic assistant and should answer user questions rigorously."),
                HumanMessage(content=str(st.session_state.translate_messages[-3])),
                AIMessage(content=str(st.session_state.translate_messages[-2])),
                HumanMessage(content=question),
            ]
        )
    else:
        raise Exception("Wrong step value")

    match llm_name:
        case 'gpt3.5':
            llm = load_gpt()

        case 'gpt4':
            llm = load_gpt4()

        case _:
            llm = load_gpt()

    chain = _prompt | llm

    result = chain.invoke({"question": question})

    return result


col1, col2 = st.columns([1, 1], gap="medium")

if 'translate_messages' not in st.session_state:
    st.session_state.translate_messages = []

if 'markdown_text' not in st.session_state:
    st.session_state.markdown_text = ''

chat_container = col1.container(height=610, border=False)

with chat_container:
    for message in st.session_state.translate_messages:
        icon = 'logo.png' if message['role'] != 'user' else None
        with st.chat_message(message['role']):
            st.markdown(message['content'])

with col2:
    if st.session_state.markdown_text != '':
        with st.container(height=520, border=True):
            st.markdown(st.session_state.markdown_text)
        st.code(st.session_state.markdown_text, language='markdown')

if prompt := st.chat_input():
    st.session_state.translate_messages = []
    prompt = prompt.replace("\n", " ").replace("\r", "")
    chat_container.chat_message("human").write(prompt)
    st.session_state.translate_messages.append({'role': 'user', 'content': prompt})

    response = get_translate_and_conclude(prompt, st.session_state.get('TranslateLLM'), 0).content
    chat_container.chat_message("ai").write(response)
    st.session_state.translate_messages.append({'role': 'assistant', 'content': response})

    query = "接下来，请用两到四句话总结一下这段文本的内容"
    chat_container.chat_message("human").write(query)
    st.session_state.translate_messages.append({'role': 'user', 'content': query})

    conclusion = get_translate_and_conclude(prompt, st.session_state.get('TranslateLLM'), 1).content
    chat_container.chat_message("ai").write(conclusion)
    st.session_state.translate_messages.append({'role': 'assistant', 'content': conclusion})

    markdown_text = f"""{prompt}\t\r\n{response}\t\r\n> {conclusion}"""
    st.session_state.markdown_text = markdown_text

    st.rerun()
