from typing import List

from langchain.chains import RetrievalQA, LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import MultiQueryRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import milvus
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

import streamlit as st

from Config import config
from llm.AgentCore import translate_sentence
from llm.ModelCore import load_gpt, load_gpt_16k
from llm.Template import RETRIEVER, ASK, TRANSLATE_TO_EN


class QuestionList(BaseModel):
    answer: List[str] = Field(description='List of generated questions.')


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=QuestionList)

    def parse(self, text: str) -> QuestionList:
        lines = text.strip().split('\n')
        return QuestionList(answer=lines)


@st.cache_resource(show_spinner='Loading Vector Database...')
def load_vectorstore():
    milvus_cfg = config.milvus_config

    model = milvus_cfg.get_model()
    collection = milvus_cfg.get_collection().NAME

    embedding = HuggingFaceBgeEmbeddings(
        model_name=model,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

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
        collection_name=collection,
        connection_args=connection_args,
        search_params={'ef': 15}
    )

    return vector_db


@st.cache_resource(show_spinner='Building retriever...')
def load_retriever():
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

    parser = LineListOutputParser()

    llm_chain = LLMChain(
        llm=retriever_llm,
        prompt=query_prompt,
        output_parser=parser
    )

    retriever = MultiQueryRetriever(
        retriever=base_retriever,
        llm_chain=llm_chain,
        parser_key='answer'
    )

    return retriever


@st.cache_data(show_spinner='Asking from LLM chain...')
def get_answer(question: str):
    prompt = PromptTemplate.from_template(ASK)
    llm = load_gpt_16k()
    retriever = load_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    question_en = translate_sentence(question, TRANSLATE_TO_EN).trans
    result = qa_chain.invoke({'query': question_en})

    return result
