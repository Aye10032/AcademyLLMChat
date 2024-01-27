import httpx
import streamlit as st
from langchain_openai import ChatOpenAI

from Config import config


@st.cache_resource
def load_gpt_16k():
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


@st.cache_resource
def load_gpt():
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
