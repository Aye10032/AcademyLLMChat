from langchain.chains import RetrievalQA
from langchain_community.vectorstores import milvus
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.prompts import PromptTemplate

import streamlit as st
from langchain_core.vectorstores import VectorStore

from Config import config
from llm.AgentCore import translate_sentence
from llm.ModelCore import load_gpt, load_gpt_16k, load_embedding_en, load_embedding_zh
from llm.RetrieverCore import multi_query_retriever, base_retriever
from llm.Template import ASK, TRANSLATE_TO_EN
from llm.storage.SqliteStore import SqliteDocStore


@st.cache_resource(show_spinner='Loading Vector Database...')
def load_vectorstore() -> VectorStore:
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
        search_params={'ef': 15}
    )

    return vector_db


@st.cache_data(show_spinner='Asking from LLM chain...')
def get_answer(question: str):
    vec_store = load_vectorstore()
    doc_store = SqliteDocStore(
        connection_string=config.get_sqlite_path()
    )
    b_retriever = base_retriever(vec_store, doc_store)
    retriever = multi_query_retriever(b_retriever)

    prompt = PromptTemplate.from_template(ASK)
    llm = load_gpt_16k()

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    # question_en = translate_sentence(question, TRANSLATE_TO_EN).trans
    result = qa_chain.invoke({'query': question})

    return result
