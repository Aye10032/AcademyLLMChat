from typing import List, Optional, Dict, Any

import streamlit as st
from langchain.chains import LLMChain
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import ParentDocumentRetriever, MultiQueryRetriever, SelfQueryRetriever, MultiVectorRetriever
from langchain.retrievers.multi_query import LineListOutputParser
from langchain.retrievers.multi_vector import SearchType
from langchain.retrievers.self_query.milvus import MilvusTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from loguru import logger

from llm.ModelCore import load_gpt
from llm.Template import GENERATE_QUESTION
from llm.storage.SqliteStore import SqliteBaseStore


class ReferenceRetriever(MultiVectorRetriever):

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        vec_doc = self.vectorstore.similarity_search(query, k=3)

        ids = []
        for doc in vec_doc:
            ref_list: list[str] = doc.metadata['ref'].split(',') if doc.metadata['ref'] != '' else []
            if ref_list:
                expr = "[" + ",".join(['"' + item + '"' for item in ref_list]) + "]"
                sub_docs: list[Document] = self.vectorstore.similarity_search(doc.page_content, expr=f'doi in {expr}')

                for d in sub_docs:
                    if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                        ids.append(d.metadata[self.id_key])

        result = self.docstore.mget(ids)

        return result


class MultiVectorSelfQueryRetriever(SelfQueryRetriever):
    doc_store: BaseStore[str, Document]
    id_key: str = "doc_id"

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        structured_query = self.query_constructor.invoke(
            {"query": query}, config={"callbacks": run_manager.get_child()}
        )
        if self.verbose:
            logger.debug(f"Generated Query: {structured_query}")

        new_query, search_kwargs = self._prepare_query(query, structured_query)
        search_kwargs['k'] = 5
        search_kwargs['fetch_k'] = 10
        logger.debug(new_query)
        logger.debug(search_kwargs)
        sub_doc = self._get_docs_with_query(new_query, search_kwargs)

        ids = []
        for d in sub_doc:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])

        result = self.doc_store.mget(ids)

        return result

    def _get_docs_with_query(
            self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        docs = self.vectorstore.search(query, 'mmr', **search_kwargs)
        return docs


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
        search_type=SearchType.mmr,
        search_kwargs={'k': 5, 'fetch_k': 10}
    )

    return retriever


@st.cache_resource(show_spinner='Building retriever...')
def multi_query_retriever(_base_retriever) -> MultiQueryRetriever:
    retriever_llm = load_gpt()
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template=GENERATE_QUESTION,
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
        include_original=False
    )

    return retriever


@st.cache_resource(show_spinner='Building retriever...')
def self_query_retriever(_vector_store: VectorStore, _doc_store: SqliteBaseStore):
    metadata_field_info = [
        AttributeInfo(
            name='title',
            description='Title of the article',
            type='string'
        ),
        AttributeInfo(
            name='section',
            description="""Title of article section.
            Among them, the Abstract chapter is usually the summary information of the whole article, 
            the Introduction chapter is usually the introduction of the research background of the article, 
            and the Conclusion chapter is usually the summary and outlook of the whole research""",
            type='string'
        ),
        AttributeInfo(
            name='year',
            description='Years in which the article was published',
            type='integer'
        ),
        AttributeInfo(
            name='author',
            description="the article's Author",
            type='integer'
        ),
        AttributeInfo(
            name='doi',
            description="The article's DOI number",
            type='string'
        )
    ]

    document_content_description = 'Specifics of the article'

    retriever_llm = load_gpt()
    retriever = MultiVectorSelfQueryRetriever.from_llm(
        llm=retriever_llm,
        vectorstore=_vector_store,
        doc_store=_doc_store,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        structured_query_translator=MilvusTranslator(),
        verbose=True,
    )

    return retriever


def reference_retriever(_vector_store: Milvus, _doc_store: SqliteBaseStore) -> ReferenceRetriever:
    retriever = ReferenceRetriever(
        vectorstore=_vector_store,
        docstore=_doc_store
    )

    return retriever
