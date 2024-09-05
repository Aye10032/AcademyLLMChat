from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from llm.ModelCore import load_gpt4o_mini
from llm.ToolCore import VecstoreSearchTool, WebSearchTool


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


def write_paper(_chat_history: ChatMessageHistory | StreamlitChatMessageHistory, question: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content='你是一个科研工作者，正在进行一些科研文献和基金项目申请的工作，因此你的写作风格需要符合科研文献的一般风格。'
                        '当需要查询信息时，优先使用向量数据库进行搜索，仅当使用者要求联网搜索时才调用工具进行联网查询。'),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{input}'),
        ]
    )

    retrieve_tool = VecstoreSearchTool(target_collection='temp1')
    web_tool = WebSearchTool()
    tools = [retrieve_tool, web_tool]

    llm = load_gpt4o_mini()
    llm_with_tools = llm.bind_tools(tools)

    history_chain = prompt | llm_with_tools

    ai_msg = history_chain.invoke({
        'chat_history': _chat_history.messages,
        'input': question,
    })

    if ai_msg.tool_calls:
        _chat_history.add_message(ai_msg)
        for tool_call in ai_msg.tool_calls:
            selected_tool = {
                'search_from_vecstore': retrieve_tool,
                'search_from_web': web_tool,
            }[tool_call["name"].lower()]
            tool_output = selected_tool.invoke(tool_call["args"])
            _chat_history.add_message(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

        result = history_chain.stream({
            'chat_history': _chat_history.messages,
            'input': question,
        })

        return result
    else:
        yield ai_msg
