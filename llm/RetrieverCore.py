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
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from loguru import logger

from llm.EmbeddingCore import Bgem3Embeddings
from llm.ModelCore import load_gpt
from llm.Template import GENERATE_QUESTION
from llm.storage.SqliteStore import SqliteBaseStore


def unique_doc(docs: List[Document]) -> List[Document]:
    result = []
    for doc in docs:
        if doc not in result:
            result.append(doc)

    return result


class ScoreRetriever(MultiVectorRetriever):
    embedding: Bgem3Embeddings

    multi_query: bool = False
    llm_chain: LLMChain

    top_k: int = 5

    def generate_queries(
            self, question: str, run_manager: CallbackManagerForRetrieverRun
    ) -> List[str]:
        response = self.llm_chain(
            {"question": question}, callbacks=run_manager.get_child()
        )
        lines = response["text"]
        return lines

    def retrieve_documents(
            self, queries: List[str], run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        documents = []
        for query in queries:
            if self.search_type == SearchType.similarity:
                short_doc: List[Document] = self.vectorstore.similarity_search(query, **self.search_kwargs)
            else:
                short_doc: List[Document] = self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
            documents.extend(short_doc)

        return documents

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.multi_query:
            queries = self.generate_queries(query, run_manager)
            queries.append(query)
            short_doc = self.retrieve_documents(queries, run_manager)
            short_doc = unique_doc(short_doc)
        else:
            if self.search_type == SearchType.similarity:
                short_doc: List[Document] = self.vectorstore.similarity_search(query, **self.search_kwargs)
            else:
                short_doc: List[Document] = self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)

        ids = []
        for sentence in short_doc:
            if self.id_key in sentence.metadata and sentence.metadata[self.id_key] not in ids:
                ids.append(sentence.metadata[self.id_key])

        docs = self.docstore.mget(ids)

        rerank_docs = self.embedding.compress_documents(docs, query)

        return rerank_docs[:self.top_k]

    async def agenerate_queries(
            self, question: str, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[str]:
        response = await self.llm_chain.acall(
            inputs={"question": question}, callbacks=run_manager.get_child()
        )
        lines = response["text"]

        return lines


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


class ExprRetriever(MultiVectorRetriever):
    embedding: Bgem3Embeddings
    expr_statement: str

    top_k: int = 5

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == SearchType.similarity:
            short_doc: List[Document] = self.vectorstore.similarity_search(
                query,
                expr=self.expr_statement,
                **self.search_kwargs
            )
        else:
            short_doc: List[Document] = self.vectorstore.max_marginal_relevance_search(
                query,
                expr=self.expr_statement,
                **self.search_kwargs
            )

        ids = []
        for sentence in short_doc:
            if self.id_key in sentence.metadata and sentence.metadata[self.id_key] not in ids:
                ids.append(sentence.metadata[self.id_key])

        docs = self.docstore.mget(ids)

        rerank_docs = self.embedding.compress_documents(docs, query)

        return rerank_docs[:self.top_k]


class MultiVectorSelfQueryRetriever(SelfQueryRetriever):
    embedding: Bgem3Embeddings
    doc_store: BaseStore[str, Document]
    id_key: str = "doc_id"
    top_k: int = 5

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        structured_query = self.query_constructor.invoke(
            {"query": query}, config={"callbacks": run_manager.get_child()}
        )

        new_query, search_kwargs = self._prepare_query(query, structured_query)
        search_kwargs['k'] = 5
        search_kwargs['fetch_k'] = 10
        sub_doc = self._get_docs_with_query(new_query, search_kwargs)

        ids = []
        for d in sub_doc:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])

        result = self.doc_store.mget(ids)

        rerank_docs = self.embedding.compress_documents(result, query)

        return rerank_docs[:self.top_k]

    def _get_docs_with_query(
            self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        docs = self.vectorstore.similarity_search(query, **search_kwargs)
        return docs


def insert_retriever(_vector_store: VectorStore, _doc_store: SqliteBaseStore) -> ParentDocumentRetriever:
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
    )

    return retriever


def base_retriever(
        _vector_store: VectorStore,
        _doc_store: SqliteBaseStore,
        _embedding: Bgem3Embeddings
) -> ScoreRetriever:
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

    retriever = ScoreRetriever(
        vectorstore=_vector_store,
        docstore=_doc_store,
        embedding=_embedding,
        multi_query=True,
        llm_chain=llm_chain,
        search_type=SearchType.similarity,
        search_kwargs={'k': 8, 'fetch_k': 10},
        top_k=5
    )

    return retriever


def self_query_retriever(
        _vector_store: VectorStore,
        _doc_store: SqliteBaseStore,
        _embedding: Bgem3Embeddings
) -> MultiVectorSelfQueryRetriever:
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
        embedding=_embedding,
        top_k=5
    )

    return retriever


def reference_retriever(_vector_store: Milvus, _doc_store: SqliteBaseStore) -> ReferenceRetriever:
    retriever = ReferenceRetriever(
        vectorstore=_vector_store,
        docstore=_doc_store
    )

    return retriever


def get_expr(fuzzy: bool = False, **kwargs) -> str:
    keys = kwargs.keys()
    expr_list = []
    if fuzzy:
        if kwargs.__contains__('title'):
            title = kwargs.pop('title')
            expr_list.append(f'title like "{title}"')

    if 'year' in keys:
        year = kwargs.pop('year')
        expr_list.append(f'year == {year}')

    for key in keys:
        value = kwargs.get(key)
        expr_list.append(f'{key} == "{value}"')

    return ' and '.join(f'({expr})' for expr in expr_list)


def expr_retriever(
        _vector_store: Milvus,
        _doc_store: SqliteBaseStore,
        _embedding: Bgem3Embeddings,
        expr_stmt: str
) -> ExprRetriever:
    retriever = ExprRetriever(
        vectorstore=_vector_store,
        docstore=_doc_store,
        embedding=_embedding,
        expr_statement=expr_stmt,
        search_type=SearchType.similarity,
        search_kwargs={'k': 8, 'fetch_k': 10},
        top_k=5
    )

    return retriever
