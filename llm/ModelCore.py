import httpx
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatZhipuAI
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
embd_cfg = config.embedding_config


@st.cache_resource(show_spinner=f'Loading {embd_cfg.model}...')
def load_embedding_en() -> Bgem3Embeddings:
    model = embd_cfg.model

    embedding = Bgem3Embeddings(
        model_name=model,
        model_kwargs={
            'device': 'cuda',
            'normalize_embeddings': embd_cfg.normalize_embeddings,
            'use_fp16': embd_cfg.fp16
        }
    )

    return embedding


@st.cache_resource(show_spinner='Loading GPT3.5 16k...')
def load_gpt_16k() -> ChatOpenAI:
    if config.openai_config.use_proxy:
        http_client = httpx.Client(proxies=config.get_proxy())
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k",
                         http_client=http_client,
                         temperature=0,
                         openai_api_key=config.openai_config.api_key)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k",
                         temperature=0,
                         openai_api_key=config.openai_config.api_key)
    return llm


@st.cache_resource(show_spinner='Loading GPT3.5...')
def load_gpt() -> ChatOpenAI:
    if config.openai_config.use_proxy:
        http_client = httpx.Client(proxies=config.get_proxy())
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                         http_client=http_client,
                         # openai_proxy=config.get_proxy(),
                         temperature=0,
                         openai_api_key=config.openai_config.api_key)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                         temperature=0,
                         openai_api_key=config.openai_config.api_key)
    return llm


@st.cache_resource(show_spinner='Loading GPT4...')
def load_gpt4() -> ChatOpenAI:
    if config.openai_config.use_proxy:
        http_client = httpx.Client(proxies=config.get_proxy())
        llm = ChatOpenAI(model_name="gpt-4",
                         http_client=http_client,
                         # openai_proxy=config.get_proxy(),
                         temperature=0,
                         openai_api_key=config.openai_config.api_key)
    else:
        llm = ChatOpenAI(model_name="gpt-4",
                         temperature=0,
                         openai_api_key=config.openai_config.api_key)
    return llm


@st.cache_resource(show_spinner='Loading Claude3...')
def load_claude3() -> ChatAnthropic:
    if config.claude_config.use_proxy:
        http_client = httpx.Client(proxies=config.get_proxy())
        llm = ChatAnthropic(temperature=0,
                            http_client=http_client,
                            anthropic_api_key=config.claude_config.api_key,
                            model_name=config.claude_config.model)
    else:
        llm = ChatAnthropic(temperature=0,
                            anthropic_api_key=config.claude_config.api_key,
                            model_name=config.claude_config.model)

    return llm


@st.cache_resource(show_spinner='Loading Qianfan...')
def load_qianfan() -> QianfanChatEndpoint:
    llm = QianfanChatEndpoint(
        model=config.qianfan_config.model,
        qianfan_ak=config.qianfan_config.api_key,
        qianfan_sk=config.qianfan_config.secret_key,
        temperature=0.05
    )

    return llm


@st.cache_resource(show_spinner='Loading Moonshot...')
def load_moonshot() -> Moonshot:
    llm = Moonshot(
        model=config.moonshot_config.model,
        moonshot_api_key=config.moonshot_config.api_key,
        temperature=0,
        max_tokens=4096
    )

    return llm


@st.cache_resource(show_spinner='Loading GLM...')
def load_glm() -> ChatZhipuAI:
    llm = ChatZhipuAI(
        api_key=config.zhipu_config.api_key,
        model=config.zhipu_config.model,
        temperature=0.05,
    )

    return llm
