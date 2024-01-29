from typing import List

from langchain.chains import RetrievalQA, LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import MultiQueryRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus, milvus
from langchain_core.globals import set_debug
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

import streamlit as st

from Config import config
from llm.ModelCore import load_gpt_16k, load_gpt
from llm.Template import ASK, RETRIEVER


class QuestionList(BaseModel):
    answer: List[str] = Field(description='List of generated questions.')


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=QuestionList)

    def parse(self, text: str) -> QuestionList:
        lines = text.strip().split('\n')
        return QuestionList(answer=lines)


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
    set_debug(True)

    llm = load_gpt_16k()
    retriever_llm = load_gpt()
    db = load_vectorstore()

    base_retriever = db.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'fetch_k': 10}
    )

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template=RETRIEVER,
    )

    llm_chain = LLMChain(
        llm=retriever_llm,
        prompt=query_prompt,
        output_parser=LineListOutputParser()
    )

    retriever = MultiQueryRetriever(
        retriever=base_retriever,
        llm_chain=llm_chain,
        parser_key='answer'
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
