from typing import List, Optional, Dict, Any, Tuple

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import ParentDocumentRetriever, MultiQueryRetriever, SelfQueryRetriever, MultiVectorRetriever
from langchain.retrievers.multi_query import LineListOutputParser
from langchain.retrievers.multi_vector import SearchType
from langchain_community.query_constructors.milvus import MilvusTranslator
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from llm.EmbeddingCore import Bgem3Embeddings
from llm.ModelCore import load_gpt4o_mini, load_glm
from llm.Template import GENERATE_QUESTION_EN, GENERATE_QUESTION_ZH
from llm.storage.SqliteStore import SqliteBaseStore

import streamlit as st


def unique_doc(docs: List[Document]) -> List[Document]:
    result = []
    for doc in docs:
        if doc not in result:
            result.append(doc)

    return result


def get_parent_id(docs: List[Document], id_key: str) -> Tuple[List, Dict]:
    ids = []
    id_map = {}
    for sentence in docs:
        if id_key in sentence.metadata:
            _id = sentence.metadata[id_key]
            if _id not in ids:
                ids.append(_id)
                id_map[_id] = [sentence.page_content]
            else:
                _temp: List = id_map.get(_id)
                _temp.append(sentence.page_content)
                id_map[_id] = _temp

    return ids, id_map


class ScoreRetriever(MultiVectorRetriever):
    embedding: Bgem3Embeddings

    multi_query: bool = False
    llm_chain: Runnable

    top_k: int = 5

    def generate_queries(
            self, question: str, run_manager: CallbackManagerForRetrieverRun
    ) -> List[str]:
        response = self.llm_chain.invoke(
            {"question": question}, config={"callbacks": run_manager.get_child()}
        )

        return response

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

        ids, id_map = get_parent_id(short_doc, self.id_key)

        docs = self.docstore.mget(ids)
        logger.info(f'retrieve {len(docs)} documents, reranking...')

        try:
            rerank_docs = self.embedding.compress_documents(docs, query)[:self.top_k]

            for i in range(len(rerank_docs)):
                context_id = rerank_docs[i].metadata[self.id_key]
                rerank_docs[i].metadata['refer_sentence'] = id_map.get(context_id) if context_id in id_map else []

            return rerank_docs
        except Exception as e:
            logger.error(f'catch exception {e} while check {ids}')

    async def agenerate_queries(
            self, question: str, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[str]:
        response = await self.llm_chain.ainvoke(
            {"question": question}, callbacks=run_manager.get_child()
        )

        return response


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

        ids, id_map = get_parent_id(short_doc, self.id_key)

        docs = self.docstore.mget(ids)
        logger.info(f'retrieve {len(docs)} documents, reranking...')

        try:
            rerank_docs = self.embedding.compress_documents(docs, query)[:self.top_k]

            for i in range(len(rerank_docs)):
                context_id = rerank_docs[i].metadata[self.id_key]
                rerank_docs[i].metadata['refer_sentence'] = id_map.get(context_id)

            return rerank_docs
        except Exception as e:
            logger.error(f'catch exception {e} while check {ids}')


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
        short_doc = self._get_docs_with_query(new_query, search_kwargs)

        ids, id_map = get_parent_id(short_doc, self.id_key)

        docs = self.docstore.mget(ids)
        logger.info(f'retrieve {len(docs)} documents, reranking...')

        try:
            rerank_docs = self.embedding.compress_documents(docs, query)[:self.top_k]

            for i in range(len(rerank_docs)):
                context_id = rerank_docs[i].metadata[self.id_key]
                rerank_docs[i].metadata['refer_sentence'] = id_map.get(context_id)

            return rerank_docs
        except Exception as e:
            logger.error(f'catch exception {e} while check {ids}')

    def _get_docs_with_query(
            self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        docs = self.vectorstore.similarity_search(query, **search_kwargs)
        return docs


def insert_retriever(_vector_store: VectorStore, _doc_store: SqliteBaseStore, language: str = 'en') -> ParentDocumentRetriever:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separators=['\n\n', '\n'],
        keep_separator=False
    )

    if language == 'en':
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
            separators=['.', '\n\n', '\n'],
            keep_separator=False
        )
    elif language == 'zh':
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
            separators=['。', '？', '\n\n', '\n'],
            keep_separator=False
        )
    else:
        raise Exception(f'wrong language type "{language}"')

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
    if st.session_state.get('app_is_zh_collection'):
        retriever_llm = load_glm()
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template=GENERATE_QUESTION_ZH,
        )
    else:
        retriever_llm = load_gpt4o_mini()
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template=GENERATE_QUESTION_EN,
        )

    parser = LineListOutputParser()

    llm_chain = query_prompt | retriever_llm | parser

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

    retriever_llm = load_gpt4o_mini()
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
        embedding: Bgem3Embeddings,
        expr_stmt: str
) -> ExprRetriever:
    retriever = ExprRetriever(
        vectorstore=_vector_store,
        docstore=_doc_store,
        embedding=embedding,
        expr_statement=expr_stmt,
        search_type=SearchType.similarity,
        search_kwargs={'k': 8, 'fetch_k': 10},
        top_k=5
    )

    return retriever
