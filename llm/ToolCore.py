from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

import streamlit as st

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
    """Retrieves documents from a vector store based on a query.

    AI在写作过程中，如遇到不明确的知识点或术语，将调用数据库进行查询以获取相关信息。

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
