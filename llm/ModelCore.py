import httpx
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.moonshot import Moonshot
from langchain_openai import ChatOpenAI

from Config import Config
from llm.EmbeddingCore import Bgem3Embeddings
from uicomponent.StatusBus import get_config

config = get_config()
if config is None:
    config = Config()

milvus_cfg = config.milvus_config


@st.cache_resource(show_spinner=f'Loading {milvus_cfg.EN_MODEL}...')
def load_embedding_en() -> Bgem3Embeddings:
    model = milvus_cfg.EN_MODEL

    embedding = Bgem3Embeddings(
        model_name=model,
        model_kwargs={
            'device': 'cuda',
            'normalize_embeddings': True,
            'use_fp16': True
        }
    )

    return embedding


@st.cache_resource(show_spinner=f'Loading {milvus_cfg.ZH_MODEL}...')
def load_embedding_zh() -> HuggingFaceEmbeddings:
    model = milvus_cfg.ZH_MODEL

    embedding = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    return embedding


@st.cache_resource(show_spinner='Loading GPT3.5 16k...')
def load_gpt_16k() -> ChatOpenAI:
    if config.openai_config.USE_PROXY:
        http_client = httpx.Client(proxies=config.get_proxy())
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
        http_client = httpx.Client(proxies=config.get_proxy())
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                         # http_client=http_client,
                         openai_proxy=config.get_proxy(),
                         temperature=0,
                         openai_api_key=config.openai_config.API_KEY)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                         temperature=0,
                         openai_api_key=config.openai_config.API_KEY)
    return llm


@st.cache_resource(show_spinner='Loading GPT4...')
def load_gpt4() -> ChatOpenAI:
    if config.openai_config.USE_PROXY:
        http_client = httpx.Client(proxies=config.get_proxy())
        llm = ChatOpenAI(model_name="gpt-4",
                         # http_client=http_client,
                         openai_proxy=config.get_proxy(),
                         temperature=0,
                         openai_api_key=config.openai_config.API_KEY)
    else:
        llm = ChatOpenAI(model_name="gpt-4",
                         temperature=0,
                         openai_api_key=config.openai_config.API_KEY)
    return llm


@st.cache_resource(show_spinner='Loading Claude3...')
def load_claude3() -> ChatAnthropic:
    if config.claude_config.USE_PROXY:
        http_client = httpx.Client(proxies=config.get_proxy())
        llm = ChatAnthropic(temperature=0,
                            http_client=http_client,
                            anthropic_api_key=config.claude_config.API_KEY,
                            model_name=config.claude_config.MODEL)
    else:
        llm = ChatAnthropic(temperature=0,
                            anthropic_api_key=config.claude_config.API_KEY,
                            model_name=config.claude_config.MODEL)

    return llm


@st.cache_resource(show_spinner='Loading Qianfan...')
def load_qianfan() -> QianfanChatEndpoint:
    llm = QianfanChatEndpoint(
        model=config.qianfan_config.MODEL,
        qianfan_ak=config.qianfan_config.API_KEY,
        qianfan_sk=config.qianfan_config.SECRET_KEY,
        temperature=0.05
    )

    return llm


@st.cache_resource(show_spinner='Loading Moonshot...')
def load_moonshot() -> Moonshot:
    llm = Moonshot(
        model=config.moonshot_config.MODEL,
        moonshot_api_key=config.moonshot_config.API_KEY,
        temperature=0,
        max_tokens=4096
    )

    return llm
