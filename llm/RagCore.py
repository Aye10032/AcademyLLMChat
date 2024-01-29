import logging

from langchain.chains import RetrievalQA
from langchain.retrievers import MultiQueryRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus, milvus
from langchain_core.prompts import PromptTemplate
import streamlit as st

from Config import config
from llm.ModelCore import load_gpt_16k, load_gpt
from llm.Template import ASK


@st.cache_resource
def load_vectorstore():
    milvus_cfg = config.milvus_config

    model_name = 'BAAI/bge-large-en-v1.5'
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vector_db: milvus = Milvus(
        embedding,
        collection_name=milvus_cfg.COLLECTION_NAME,
        connection_args={'host': milvus_cfg.MILVUS_HOST, 'port': milvus_cfg.MILVUS_PORT},
        search_params={'ef': 15}
    )

    return vector_db


@st.cache_resource
def get_qa_chain():
    llm = load_gpt_16k()
    llm_retriever = load_gpt()
    db = load_vectorstore()

    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 5, 'fetch_k': 15}
        ),
        llm=llm_retriever,
        include_original=True
    )
    qa_chain_prompt = PromptTemplate.from_template(ASK)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': qa_chain_prompt}
    )

    return qa_chain


@st.cache_data
def ask_from_rag(question: str):
    qa_chain = get_qa_chain()
    result = qa_chain.invoke({'query': question})

    return result
