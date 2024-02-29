from typing import List, Optional, Dict, Any

import streamlit as st
from langchain.chains import LLMChain
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import ParentDocumentRetriever, MultiQueryRetriever, SelfQueryRetriever, MultiVectorRetriever
from langchain.retrievers.self_query.milvus import MilvusTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.callbacks import Callbacks, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from llm.ModelCore import load_gpt
from llm.Template import RETRIEVER
from llm.storage.SqliteStore import SqliteBaseStore


class QuestionList(BaseModel):
    answer: List[str] = Field(description='List of generated questions.')


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=QuestionList)

    def parse(self, text: str) -> QuestionList:
        lines = text.strip().split('\n')
        return QuestionList(answer=lines)


class ReferenceRetriever(MultiVectorRetriever):
    retriever: SelfQueryRetriever

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.retriever.invoke(query)


@st.cache_resource(show_spinner='Building base retriever...')
def base_retriever(_vector_store: VectorStore, _doc_store: SqliteBaseStore) -> ParentDocumentRetriever:
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
def multi_query_retriever(_base_retriever) -> MultiQueryRetriever:
    retriever_llm = load_gpt()
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
        retriever=_base_retriever,
        llm_chain=llm_chain,
        parser_key='answer',
        include_original=False
    )

    return retriever


@st.cache_resource(show_spinner='Building retriever...')
def load_self_query_retriever(_vector_store: VectorStore):
    metadata_field_info = [
        AttributeInfo(
            name='title',
            description='Title of the article',
            type='string'
        ),
        AttributeInfo(
            name='section',
            description='Title of article section',
            type='string'
        ),
        AttributeInfo(
            name='year',
            description='Years in which the article was published',
            type='integer'
        ),
        AttributeInfo(
            name='doi',
            description='The article\'s DOI number',
            type='string'
        ),
        AttributeInfo(
            name='ref',
            description='The DOI numbers of the articles cited in this text, separated by ","',
            type='string'
        ),
    ]

    document_content_description = 'Specifics of the article'

    retriever_llm = load_gpt()
    retriever = SelfQueryRetriever.from_llm(
        llm=retriever_llm,
        vectorstore=_vector_store,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        structured_query_translator=MilvusTranslator()
    )

    return retriever
