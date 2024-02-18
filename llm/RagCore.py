from typing import List

from langchain.chains import RetrievalQA, LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import MultiQueryRetriever, ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import milvus
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

import streamlit as st

from Config import config
from llm.AgentCore import translate_sentence
from llm.ModelCore import load_gpt, load_gpt_16k, load_embedding_en, load_embedding_zh
from llm.Template import RETRIEVER, ASK, TRANSLATE_TO_EN
from storage.SqliteStore import SqliteDocStore


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
def load_retriever():
    retriever_llm = load_gpt()
    vector_store = load_vectorstore()
    doc_store = SqliteDocStore(
        connection_string=config.get_sqlite_path()
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separators=['\n\n', '\n'],
        keep_separator=False
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        separators=['.', '\n\n', '\n'],
        keep_separator=False
    )

    base_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
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
        parser_key='answer',
        include_original=True
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
