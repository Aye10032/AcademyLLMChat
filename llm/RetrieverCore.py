from typing import List

import streamlit as st
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import ParentDocumentRetriever, MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from llm.ModelCore import load_gpt
from llm.Template import RETRIEVER


class QuestionList(BaseModel):
    answer: List[str] = Field(description='List of generated questions.')


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=QuestionList)

    def parse(self, text: str) -> QuestionList:
        lines = text.strip().split('\n')
        return QuestionList(answer=lines)


@st.cache_resource(show_spinner='Building base retriever...')
def base_retriever(_vector_store, _doc_store) -> ParentDocumentRetriever:
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

    retriever = ParentDocumentRetriever(
        vectorstore=_vector_store,
        docstore=_doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type='mmr',
        search_kwargs={'k': 5, 'fetch_k': 10}
    )

    return retriever


@st.cache_resource(show_spinner='Building retriever...')
def multi_query_retriever() -> MultiQueryRetriever:
    retriever_llm = load_gpt()
    b_retriever = base_retriever()
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
        retriever=b_retriever,
        llm_chain=llm_chain,
        parser_key='answer',
        include_original=True
    )

    return retriever
