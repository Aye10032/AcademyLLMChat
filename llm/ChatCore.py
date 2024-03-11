from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from llm.ModelCore import load_gpt

import streamlit as st


@st.cache_data(show_spinner='chat with GPT...')
def chat_with_history(_chat_history: ChatMessageHistory | StreamlitChatMessageHistory, question: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                'You are a helpful assistant. Answer all questions to the best of your ability.',
            ),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{input}'),
        ]
    )
    llm = load_gpt()
    chain = prompt | llm

    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: _chat_history,
        input_messages_key='input',
        history_messages_key='chat_history',
    )

    result = chain_with_message_history.invoke(
        {'input': question},
        {'configurable': {'session_id': 'unused'}},
    )

    return result
