from operator import itemgetter
from typing import List

from langchain_community.vectorstores import milvus
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

import streamlit as st
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from Config import config
from llm.AgentCore import translate_sentence
from llm.ModelCore import load_gpt_16k, load_embedding_en, load_embedding_zh
from llm.RetrieverCore import multi_query_retriever, base_retriever, self_query_retriever
from llm.Template import TRANSLATE_TO_EN, ASK_SYSTEM, ASK_USER
from llm.storage.SqliteStore import SqliteDocStore


class CitedAnswer(BaseModel):
    """Answer the user question both in English and Chinese based only on the given essay fragment, and cite the sources used."""

    answer_en: str = Field(
        ...,
        description="The answer to the user question in English, which is based only on the given fragment.",
    )
    answer_zh: str = Field(
        ...,
        description="The answer to the user question in Chinese, which is based only on the given fragment.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC fragment which justify the answer.",
    )


def format_docs(docs: List[Document]) -> str:
    formatted = [
        f"""Fragment ID: {i}
        Essay Title: {doc.metadata['title']}
        Essay Author: {doc.metadata['author']}
        Publish year: {doc.metadata['year']}
        Essay DOI: {doc.metadata['doi']}
        Fragment Snippet: {doc.page_content}
        """
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)


@st.cache_resource(show_spinner='Loading Vector Database...')
def load_vectorstore(collection_name: str) -> Milvus:
    milvus_cfg = config.milvus_config

    if milvus_cfg.get_collection().LANGUAGE == 'zh':
        embedding = load_embedding_zh()
    else:
        embedding = load_embedding_en()

    if milvus_cfg.USING_REMOTE:
        connection_args = {
            'uri': milvus_cfg.REMOTE_DATABASE['url'],
            'user': milvus_cfg.REMOTE_DATABASE['username'],
            'password': milvus_cfg.REMOTE_DATABASE['password'],
            'secure': True,
        }
    else:
        connection_args = {
            'host': milvus_cfg.MILVUS_HOST,
            'port': milvus_cfg.MILVUS_PORT
        }

    vector_db: milvus = Milvus(
        embedding,
        collection_name=collection_name,
        connection_args=connection_args,
        search_params={'ef': 15},
        auto_id=True
    )

    return vector_db


def load_doc_store() -> SqliteDocStore:
    doc_store = SqliteDocStore(
        connection_string=config.get_sqlite_path()
    )

    return doc_store


@st.cache_data(show_spinner='Asking from LLM chain...')
def get_answer(question: str, self_query: bool = False):
    vec_store = load_vectorstore(config.milvus_config.get_collection().NAME)
    doc_store = load_doc_store()

    llm = load_gpt_16k()
    llm_tool = llm.bind_tools(
        [CitedAnswer],
        tool_choice="CitedAnswer",
    )

    question = translate_sentence(question, TRANSLATE_TO_EN).trans

    if self_query:
        retriever = self_query_retriever(vec_store, doc_store)
    else:
        b_retriever = base_retriever(vec_store, doc_store)
        retriever = multi_query_retriever(b_retriever)

    prompt = ChatPromptTemplate.from_messages([('system', ASK_SYSTEM), ('human', ASK_USER)])

    formatter = itemgetter("docs") | RunnableLambda(format_docs)

    output_parser = JsonOutputKeyToolsParser(key_name="CitedAnswer", return_single=True)
    answer = prompt | llm_tool | output_parser
    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        .assign(context=formatter)
        .assign(answer=answer)
        .pick(["answer", "docs"])
    )

    result = chain.invoke(question)

    return result
