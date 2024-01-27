from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus, milvus
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
import streamlit as st

from Config import config
from rag.Template import ASK


@st.cache_resource
def load_retrieval():
    milvus_cfg = config.milvus_config

    model_name = "BAAI/bge-large-en-v1.5"
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
        connection_args={"host": milvus_cfg.MILVUS_HOST, "port": milvus_cfg.MILVUS_PORT},
    )
    retriever = VectorStoreRetriever(vectorstore=vector_db)

    return retriever


@st.cache_resource
def load_llm():
    if config.openai_config.USE_PROXY:
        import httpx

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
def get_qa_chain():
    llm = load_llm()
    retriever = load_retrieval()
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


if __name__ == '__main__':
    data = ask_from_rag('what is npq?')
