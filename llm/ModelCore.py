import httpx
import streamlit as st
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain_community.llms.moonshot import Moonshot
from langchain_openai import ChatOpenAI

from Config import Config
from llm.EmbeddingCore import BgeM3Embeddings, BgeReranker
from uicomponent.StatusBus import get_config

config = get_config()
if config is None:
    config = Config()

milvus_cfg = config.milvus_config
embd_cfg = config.embedding_config
reranker_cfg = config.reranker_config


@st.cache_resource(show_spinner=f'Loading {embd_cfg.model}...')
def load_embedding() -> BgeM3Embeddings:
    embedding = BgeM3Embeddings(
        model_name=embd_cfg.model,
        use_fp16=embd_cfg.fp16,
        encode_kwargs={
            'normalize_embeddings': embd_cfg.normalize
        },
        local_load=embd_cfg.save_local,
        local_path=embd_cfg.local_path
    )

    return embedding


@st.cache_resource(show_spinner=f'Loading {reranker_cfg.model}...')
def load_reranker() -> BgeReranker:
    reranker = BgeReranker(
        model_name=reranker_cfg.model,
        use_fp16=reranker_cfg.fp16,
        encode_kwargs={
            'normalize': reranker_cfg.normalize
        },
        local_load=reranker_cfg.save_local,
        local_path=reranker_cfg.local_path
    )

    return reranker


@st.cache_resource(show_spinner='Loading GPT4o...')
def load_gpt4o() -> ChatOpenAI:
    if config.openai_config.use_proxy:
        http_client = httpx.Client(proxies=config.get_proxy())
        llm = ChatOpenAI(model_name="gpt-4o",
                         http_client=http_client,
                         temperature=0.4,
                         openai_api_key=config.openai_config.api_key)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k",
                         temperature=0.4,
                         openai_api_key=config.openai_config.api_key)
    return llm


@st.cache_resource(show_spinner='Loading GPT4o mini...')
def load_gpt4o_mini() -> ChatOpenAI:
    if config.openai_config.use_proxy:
        http_client = httpx.Client(proxies=config.get_proxy())
        llm = ChatOpenAI(model_name="gpt-4o-mini",
                         http_client=http_client,
                         temperature=0.4,
                         openai_api_key=config.openai_config.api_key)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                         temperature=0.4,
                         openai_api_key=config.openai_config.api_key)
    return llm


@st.cache_resource(show_spinner='Loading GPT4...')
def load_gpt4() -> ChatOpenAI:
    if config.openai_config.use_proxy:
        http_client = httpx.Client(proxies=config.get_proxy())
        llm = ChatOpenAI(model_name="gpt-4-turbo-2024-04-09",
                         http_client=http_client,
                         # openai_proxy=config.get_proxy(),
                         temperature=0.6,
                         openai_api_key=config.openai_config.api_key)
    else:
        llm = ChatOpenAI(model_name="gpt-4",
                         temperature=0.6,
                         openai_api_key=config.openai_config.api_key)
    return llm


@st.cache_resource(show_spinner='Loading GLM4-flash...')
def load_glm4_flash() -> ChatOpenAI:
    llm = ChatOpenAI(
        model=config.zhipu_config.model,
        openai_api_base='https://open.bigmodel.cn/api/paas/v4/',
        openai_api_key=config.zhipu_config.api_key,
        temperature=0.05,
    )

    return llm
