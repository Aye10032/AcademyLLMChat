from typing import List

from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain.retrievers.self_query.milvus import MilvusTranslator
from langchain_community.vectorstores import milvus
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.prompts import PromptTemplate

import streamlit as st

from Config import config
from llm.AgentCore import translate_sentence
from llm.ModelCore import load_gpt, load_gpt_16k, load_embedding_en, load_embedding_zh
from llm.RetrieverCore import multi_query_retriever
from llm.Template import ASK, TRANSLATE_TO_EN
from llm.storage.SqliteStore import SqliteDocStore


@st.cache_resource(show_spinner='Loading Vector Database...')
def load_vectorstore():
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


@st.cache_resource(show_spinner='Building retriever...')
def load_self_query_retriever():
    metadata_field_info = [
        AttributeInfo(
            name='title',
            description='Title of the article',
            type='string'
        ),
        AttributeInfo(
            name='section',
            description='Title of article section',
            type='string'
        ),
        AttributeInfo(
            name='year',
            description='Years in which the article was published',
            type='integer'
        ),
        AttributeInfo(
            name='doi',
            description='The article\'s DOI number',
            type='string'
        ),
        AttributeInfo(
            name='ref',
            description='The DOI numbers of the articles cited in this text, separated by ","',
            type='string'
        ),
    ]

    document_content_description = 'Specifics of the article'

    retriever_llm = load_gpt()
    vector_store = load_vectorstore()
    retriever = SelfQueryRetriever.from_llm(
        llm=retriever_llm,
        vectorstore=vector_store,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        structured_query_translator=MilvusTranslator()
    )

    return retriever


@st.cache_data(show_spinner='Asking from LLM chain...')
def get_answer(question: str):
    prompt = PromptTemplate.from_template(ASK)
    llm = load_gpt_16k()
    retriever = multi_query_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    # question_en = translate_sentence(question, TRANSLATE_TO_EN).trans
    result = qa_chain.invoke({'query': question})

    return result
