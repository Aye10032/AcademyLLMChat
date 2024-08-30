from typing import Any, Type, Optional

from duckduckgo_search import DDGS
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool, BaseTool, ToolException

import streamlit as st
from pydantic.v1 import BaseModel, Field

from Config import Config
from llm.ModelCore import load_embedding
from llm.RagCore import load_vectorstore, load_doc_store
from llm.RetrieverCore import base_retriever
from uicomponent.StatusBus import get_config

config: Config = get_config()


def format_docs(docs: list[Document]) -> str:
    return "\n\n-------------------------\n\n".join([doc.page_content for doc in docs])


@tool
@st.cache_data(show_spinner='Search from storage...')
def retrieve_from_vecstore(query: str) -> str:
    """从向量数据库中进行查询操作

    AI在写作过程中，如遇到不明确的知识点或术语，可以调用此工具从数据库中进行查询以获取相关信息。

    :param query: The search query to retrieve documents.
    :return: A list of documents that match the query.
    """
    embedding = load_embedding()

    collection_name = 'test'
    vec_store = load_vectorstore(collection_name, embedding)
    doc_store = load_doc_store(config.get_sqlite_path(collection_name))

    retriever = base_retriever(vec_store, doc_store, embedding)

    chain = retriever | RunnableLambda(format_docs)
    output = chain.invoke(query)

    return output


class WebSearchInput(BaseModel):
    query: str = Field(description="联网搜索的关键词")


class WebSearchTool(BaseTool):
    name = 'search_from_web'
    description = '通过搜索引擎进行联网搜索。AI可以通过调用此工具，联网查询一些自己不清楚的或者比较新的信息。'
    args_schema: Type[BaseModel] = WebSearchInput
    return_direct = False
    handle_tool_error = True

    region: str = 'wt-wt'
    max_search_result: int = 6

    @st.cache_data(show_spinner='Search from web...')
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """调用工具进行联网搜索"""
        with DDGS(proxy=config.get_proxy()) as ddgs:
            search_result = ddgs.text(
                query,
                region=self.region,
                max_results=self.max_search_result,
            )
            if search_result:
                text_list = []
                for r in search_result:
                    text_list.append(
                        f"# {r['title']}\n"
                        f"{r['body']}"
                    )
                return "\n\n-------------------------\n\n".join(text_list)

        raise ToolException("所给出的问题没有在互联网上找到相关信息。")


def main() -> None:
    search_tool = WebSearchTool()
    result = search_tool.invoke('什么是宏基因组？')
    print(result)


if __name__ == '__main__':
    main()
