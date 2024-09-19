from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from llm.ModelCore import load_gpt4o_mini
from llm.ToolCore import VecstoreSearchTool, WebSearchTool

import streamlit as st


def chat_with_history(_chat_history: BaseChatMessageHistory, question: str):
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
    llm = load_gpt4o_mini()
    chain = prompt | llm

    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: _chat_history,
        input_messages_key='input',
        history_messages_key='chat_history',
    )

    result = chain_with_message_history.stream(
        {'input': question},
        {'configurable': {'session_id': 'unused'}},
    )

    return result


def write_paper(
        _chat_history: BaseChatMessageHistory
):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content='你是一个科研工作者，正在进行一些科研文献和基金项目申请的工作，因此你的写作风格需要符合科研文献的一般风格。'
                        '当需要查询信息时，优先使用向量数据库进行搜索，仅当使用者要求联网搜索时才调用工具进行联网查询。'),
            MessagesPlaceholder(variable_name='chat_history')
        ]
    )

    retrieve_tool = VecstoreSearchTool(target_collection='temp1')
    web_tool = WebSearchTool()
    tools = [retrieve_tool, web_tool]

    llm = load_gpt4o_mini()
    llm_with_tools = llm.bind_tools(tools)

    history_chain = prompt | llm_with_tools

    messages = _chat_history.messages.copy()

    with st.spinner('正在分析您的需求...'):
        ai_msg = history_chain.invoke({'chat_history': messages})

    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = {
            'search_from_vecstore': retrieve_tool,
            'search_from_web': web_tool,
        }[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    result = history_chain.stream({'chat_history': messages})

    return result


def conclude_chat(_chat_history: BaseChatMessageHistory):
    """
    Summarize the main content of a chat conversation.

    This function generates a summary of the chat conversation, including the identities of the participants,
    the topics discussed, key points raised, and any questions or solutions proposed. The summary is concise,
    not exceeding 18 characters.

    :param _chat_history: The chat history to be summarized.
    :return: A summary of the chat conversation.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="你即将看到一段对话记录。"
                        "请总结对话的主要内容，包括对话参与者的身份、讨论的主题、提出的关键观点、问题或解决方案。"
                        "确保抓住对话中的重要细节和关键时刻，同时控制字数，不要超过18字。"
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(
                content="根据对话内容，生成一个词组或短语（**不超过18个字**）作为该对话的概览词。该概览词应能反映对话的核心主题或目的。"),
        ]
    )
    llm = load_gpt4o_mini()
    chain = prompt | llm

    result = chain.invoke({"history": _chat_history.messages})

    return result
