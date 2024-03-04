from langchain.chains import RetrievalQA
from langchain_community.vectorstores import milvus
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.prompts import PromptTemplate

import streamlit as st

from Config import config
from llm.AgentCore import translate_sentence
from llm.ModelCore import load_gpt_16k, load_embedding_en, load_embedding_zh
from llm.RetrieverCore import multi_query_retriever, base_retriever, self_query_retriever
from llm.Template import ASK, TRANSLATE_TO_EN
from llm.storage.SqliteStore import SqliteDocStore


@st.cache_resource(show_spinner='Loading Vector Database...')
def load_vectorstore() -> Milvus:
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
        collection_name=milvus_cfg.get_collection().NAME,
        connection_args=connection_args,
        search_params={'ef': 15},
        auto_id=True
    )

    return vector_db


@st.cache_resource
def load_doc_store() -> SqliteDocStore:
    doc_store = SqliteDocStore(
        connection_string=config.get_sqlite_path()
    )

    return doc_store


@st.cache_data(show_spinner='Asking from LLM chain...')
def get_answer(question: str, self_query: bool = False):
    vec_store = load_vectorstore()
    doc_store = load_doc_store()

    prompt = PromptTemplate.from_template(ASK)
    llm = load_gpt_16k()

    if self_query:
        retriever = self_query_retriever(vec_store, doc_store)
        question = translate_sentence(question, TRANSLATE_TO_EN).trans
    else:
        b_retriever = base_retriever(vec_store, doc_store)
        retriever = multi_query_retriever(b_retriever)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    result = qa_chain.invoke({'query': question})

    return result
