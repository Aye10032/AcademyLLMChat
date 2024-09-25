import json
import re
from typing import Type, Optional, Any

import requests
from duckduckgo_search import DDGS
from langchain_community.document_loaders import WebBaseLoader, AsyncHtmlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, ToolException

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel, Field

from Config import Config
from llm.ModelCore import load_reranker, load_embedding
from llm.RagCore import load_vectorstore, load_doc_store
from llm.RetrieverCore import base_retriever
from uicomponent.StatusBus import get_config

config: Config = get_config()


class VecstoreSearchInput(BaseModel):
    query: str = Field(description="用于从向量数据库中召回长文本的搜索文本")


class VecstoreSearchTool(BaseTool):
    name: str = 'search_from_vecstore'
    description: str = '根据查询文本从向量数据库中搜索相关的知识。AI在写作过程中，如遇到不明确的知识点或术语，可以调用此工具从数据库中进行查询以获取相关信息。'
    args_schema: Type[BaseModel] = VecstoreSearchInput
    return_direct: bool = False
    handle_tool_error: bool = True

    embedding: Any = None
    reranker: Any = None

    target_collection: Optional[str] = None

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> list[Document]:
        """从向量数据库中进行查询操作"""
        logger.info(f'Calling VecstoreSearchTool with query {query}')

        def format_docs(docs: list[Document]) -> str:
            return "\n\n-------------------------\n\n".join([doc.page_content for doc in docs])

        vec_store = load_vectorstore(self.target_collection, embedding)
        doc_store = load_doc_store(config.get_sqlite_path(self.target_collection))

        retriever = base_retriever(vec_store, doc_store, reranker)

        output = retriever.invoke(query)

        return output


class WebSearchInput(BaseModel):
    query: str = Field(description="联网搜索的关键词")


class WebSearchTool(BaseTool):
    name: str = 'search_from_web'
    description: str = '通过搜索引擎进行联网搜索。AI可以通过调用此工具，联网查询一些自己不清楚的或者比较新的信息。'
    args_schema: Type[BaseModel] = WebSearchInput
    return_direct: bool = False
    handle_tool_error: bool = True

    region: str = 'wt-wt'
    max_search_result: int = 6

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> list[Document]:
        """调用工具进行联网搜索"""
        logger.info(f'Calling WebSearchTool with query {query}')

        def search(_query: str):
            if serper_api := config.serper_config.api_key:
                url = "https://google.serper.dev/search"

                payload = json.dumps({
                    "q": _query,
                    "k": self.max_search_result,
                    "gl": "us",
                    "hl": "en"
                })
                headers = {
                    'X-API-KEY': serper_api,
                    'Content-Type': 'application/json',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }

                if config.serper_config.use_proxy:
                    response = requests.request(
                        "POST", url,
                        headers=headers,
                        data=payload,
                        proxies={
                            'http': config.get_proxy(),
                            'https': config.get_proxy(),
                        }
                    )
                else:
                    response = requests.request(
                        "POST", url,
                        headers=headers,
                        data=payload
                    )

                if response.status_code == 200:
                    search_data = json.loads(response.text)
                    url_list = [
                        organic['link']
                        for organic in search_data['organic']
                    ]
                    return url_list
            else:
                logger.warning('no serper api found, using ddgs instead')
                with DDGS(proxy=config.get_proxy()) as ddgs:
                    search_result = ddgs.text(
                        _query,
                        region=self.region,
                        max_results=self.max_search_result,
                    )
                    if search_result:
                        url_list = [
                            result['href']
                            for result in search_result
                        ]
                        return url_list

            raise ToolException("所给出的问题没有在互联网上找到相关信息。")

        def load_webpage(_urls: list[str]) -> list[Document]:
            loader = WebBaseLoader(
                _urls,
                header_template={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                },
                encoding='utf-8',
                proxies={
                    'http': config.get_proxy(),
                    'https': config.get_proxy()
                }
            )
            docs = loader.load()
            logger.info(f'Get {len(_urls)} pages, converting to text...')

            markdownify = MarkdownifyTransformer()
            docs_transform = markdownify.transform_documents(docs)

            for doc in docs_transform:
                doc.metadata['type'] = 'web_search'

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=450,
                chunk_overlap=0,
                separators=['\n\n'],
                keep_separator=False
            )

            docs_split = text_splitter.split_documents(docs_transform)

            return docs_split

        urls = search(query)
        return load_webpage(urls)


def main() -> None:
    search_tool = WebSearchTool()
    question = '什么是泛基因组？'
    results = search_tool.invoke(question)
    reranker = load_reranker()
    reranked_docs = reranker.compress_documents(results, question)
    for doc in reranked_docs:
        print(doc.page_content)
        print(doc.metadata['score'])
        print('==========================')


if __name__ == '__main__':
    main()
