import httpx
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from Config import config


@st.cache_resource(show_spinner=f'Loading {config.milvus_config.EN_MODEL}...')
def load_embedding_en() -> HuggingFaceBgeEmbeddings:
    model = config.milvus_config.EN_MODEL

    embedding = HuggingFaceBgeEmbeddings(
        model_name=model,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    return embedding


@st.cache_resource(show_spinner=f'Loading {config.milvus_config.ZH_MODEL}...')
def load_embedding_zh() -> HuggingFaceEmbeddings:
    model = config.milvus_config.ZH_MODEL

    embedding = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    return embedding


@st.cache_resource(show_spinner='Loading GPT3.5 16k...')
def load_gpt_16k() -> ChatOpenAI:
    if config.openai_config.USE_PROXY:
        http_client = httpx.Client(proxies=config.PROXY)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k",
                         http_client=http_client,
                         temperature=0,
                         openai_api_key=config.openai_config.API_KEY)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k",
                         temperature=0,
                         openai_api_key=config.openai_config.API_KEY)
    return llm


@st.cache_resource(show_spinner='Loading GPT3.5...')
def load_gpt() -> ChatOpenAI:
    if config.openai_config.USE_PROXY:
        http_client = httpx.Client(proxies=config.PROXY)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                         http_client=http_client,
                         temperature=0,
                         openai_api_key=config.openai_config.API_KEY)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                         temperature=0,
                         openai_api_key=config.openai_config.API_KEY)
    return llm
