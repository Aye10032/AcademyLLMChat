from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from llm.ModelCore import load_gpt4o_mini
from llm.ToolCore import VecstoreSearchTool


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
    retrieve_tool = VecstoreSearchTool()
    tools = [retrieve_tool]

    llm = load_gpt4o_mini()
    llm_with_tools = llm.bind_tools(tools)

    messages = _chat_history.messages.copy()
    messages.append(HumanMessage(question))

    ai_msg = llm_with_tools.invoke(messages)

    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"search_from_vecstore": retrieve_tool}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    result = llm_with_tools.stream(messages)

    return result
