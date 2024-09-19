from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from llm.ModelCore import load_gpt4o_mini
from llm.ToolCore import VecstoreSearchTool, WebSearchTool

import streamlit as st


@st.cache_resource(show_spinner='Build graph...')
def write_with_db(collection_name: str) -> CompiledStateGraph:
    retrieve_tool = VecstoreSearchTool(target_collection=collection_name)
    web_tool = WebSearchTool()
    tools = [retrieve_tool, web_tool]

    llm = load_gpt4o_mini()
    llm_with_tools = llm.bind_tools(tools)

    sys_msg = SystemMessage(
        content='你是一个科研工作者，正在进行一些科研文献和基金项目申请的工作，因此你的写作风格需要符合科研文献的一般风格。'
                '当需要查询信息时，优先使用向量数据库进行搜索，仅当使用者要求联网搜索时才调用工具进行联网查询。'
    )

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    builder = StateGraph(MessagesState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition, )
    builder.add_edge("tools", "assistant")

    react_graph = builder.compile()

    return react_graph
