from operator import itemgetter
from typing import List

from langchain_community.vectorstores import milvus
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

import streamlit as st
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from llm.AgentCore import translate_sentence
from llm.ModelCore import load_gpt_16k, load_embedding_en, load_embedding_zh, load_gpt4
from llm.RetrieverCore import *
from llm.Template import *
from llm.storage.SqliteStore import SqliteDocStore
from uicomponent.StatusBus import get_config

config = get_config()


class CitedAnswer(BaseModel):
    """Answer the user question both in English and Chinese based only on the given essay fragment, and cite the sources used."""

    answer_en: str = Field(
        ...,
        description='The answer to the user question in English, which is based only on the given fragment, , and use "[]" at the end of the sentence to mark the ID of the quoted fragment',
    )
    answer_zh: str = Field(
        ...,
        description="Chinese translation of English answer",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC fragment which justify the answer.",
    )


def format_docs(docs: List[Document]) -> str:
    formatted = [
        f"""Fragment ID: {i + 1}
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

    vector_db: milvus = Milvus(
        embedding,
        collection_name=collection_name,
        connection_args=milvus_cfg.get_conn_args(),
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
def get_answer(question: str, self_query: bool = False, expr_stmt: str = None, *, llm_name: str):
    vec_store = load_vectorstore(config.milvus_config.get_collection().NAME)
    embedding = load_embedding_en()
    doc_store = load_doc_store()

    if llm_name == 'gpt3.5-16k':
        llm = load_gpt_16k()
    elif llm_name == 'gpt4':
        llm = load_gpt4()
    else:
        llm = load_gpt()

    question = translate_sentence(question, TRANSLATE_TO_EN).trans

    if self_query:
        if expr_stmt is not None:
            retriever = expr_retriever(vec_store, doc_store, embedding, expr_stmt)
        else:
            retriever = self_query_retriever(vec_store, doc_store)
    else:

        retriever = base_retriever(vec_store, doc_store, embedding)

    parser = JsonOutputParser(pydantic_object=CitedAnswer)

    system_prompt = PromptTemplate(
        template=ASK_SYSTEM,
        input_variables=["format_instructions", "example_q", "example_a"],
    )

    system_str = system_prompt.format(
        format_instructions=parser.get_format_instructions(),
        example_q=EXAMPLE_Q,
        example_a=EXAMPLE_A
    )

    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_str),
         ('human', ASK_USER)]
    )

    formatter = itemgetter("docs") | RunnableLambda(format_docs)

    chain = prompt | llm | parser
    answer_chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        .assign(context=formatter)
        .assign(answer=chain)
        .pick(["answer", "docs"])
    )

    result = answer_chain.invoke(question)

    return result
