from operator import itemgetter

from langchain_milvus.vectorstores import milvus
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from llm.AgentCore import translate_sentence
from llm.EmbeddingCore import BgeM3Embeddings
from llm.ModelCore import load_gpt4o, load_embedding, load_gpt4, load_reranker
from llm.RetrieverCore import *
from llm.Template import *
from storage.SqliteStore import SqliteDocStore
from uicomponent.StatusBus import get_config

config = get_config()


class CitedAnswerEN(BaseModel):
    """Answer the user question both in English and Chinese based only on the given essay fragment, and cite the sources used."""

    answer_en: str = Field(
        ...,
        description='The answer to the user question in English, which is based only on the given fragment, , and use "[]" at the end of the sentence to mark the ID of the quoted fragment',
    )
    answer_zh: str = Field(
        ...,
        description="Chinese translation of English answer",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC fragment which justify the answer.",
    )


class CitedAnswerZH(BaseModel):
    """仅使用你被给到的文献片段来回答问题，对于回答中用到的文献片段，以学术论文引用的形式用“[]”标注出来它的ID."""

    answer_zh: str = Field(
        ...,
        description='对于问题的回答，仅使用你被给到的文献片段来生成答案。对于回答中用到的文献片段，以学术论文引用的形式用“[]”标注出来它的ID.',
    )
    citations: List[int] = Field(
        ...,
        description="你的回答中用到的文献片段的ID",
    )


def format_docs(docs: List[Document]) -> str:
    if st.session_state.get('app_is_zh_collection'):
        formatted = [(
            f"文献 ID: {i + 1}\n"
            f"文献标题: {doc.metadata['title']}\n"
            f"文献作者: {doc.metadata['author']}\n"
            f"发表年份: {doc.metadata['year']}\n"
            f"文献DOI编号: {doc.metadata['doi']}\n"
            f"文献片段内容: {doc.page_content}\n"
        )
            for i, doc in enumerate(docs)
        ]
    else:
        formatted = [(
            f"Fragment ID: {i + 1}\n"
            f"Essay Title: {doc.metadata['title']}\n"
            f"Essay Author: {doc.metadata['author']}\n"
            f"Publish year: {doc.metadata['year']}\n"
            f"Essay DOI: {doc.metadata['doi']}\n"
            f"Fragment Snippet: {doc.page_content}\n"
        )
            for i, doc in enumerate(docs)
        ]
    return "\n\n" + "\n\n----------------------------\n\n".join(formatted)


@st.cache_resource(show_spinner='Loading Vector Database...')
def load_vectorstore(collection_name: str, _embedding_model: BgeM3Embeddings) -> Milvus:
    milvus_cfg = config.milvus_config

    vector_db: milvus = Milvus(
        _embedding_model,
        collection_name=collection_name,
        connection_args=milvus_cfg.get_conn_args(),
        search_params={'ef': 15},
        auto_id=True
    )

    return vector_db


def load_doc_store(db_path: str) -> SqliteDocStore:
    doc_store = SqliteDocStore(
        connection_string=db_path
    )

    return doc_store


@st.cache_data(show_spinner='Asking from LLM chain...')
def get_answer(
        collection_name: str,
        question: str,
        self_query: bool = False,
        expr_stmt: str = None,
        *,
        llm_name: str
):
    embedding = load_embedding()
    reranker = load_reranker()

    vec_store = load_vectorstore(collection_name, embedding)
    doc_store = load_doc_store(config.get_sqlite_path(collection_name))

    if llm_name == 'gpt4o':
        llm = load_gpt4o()
    elif llm_name == 'gpt4':
        llm = load_gpt4()
    elif llm_name == 'GLM-4':
        llm = load_glm4_flash()
    else:
        llm = load_gpt4o_mini()

    if not st.session_state.get('app_is_zh_collection'):
        question = translate_sentence(question, TRANSLATE_TO_EN).trans
        parser = JsonOutputParser(pydantic_object=CitedAnswerEN)

        system_prompt = PromptTemplate(
            template=ASK_SYSTEM_EN,
            input_variables=["format_instructions", "example_q", "example_a"],
        )

        system_str = system_prompt.format(
            format_instructions=parser.get_format_instructions(),
            example_q=EXAMPLE_Q,
            example_a=EXAMPLE_A
        )

        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=system_str),
             ('human', ASK_USER_EN)]
        )
    else:
        parser = JsonOutputParser(pydantic_object=CitedAnswerZH)

        system_prompt = PromptTemplate(
            template=ASK_SYSTEM_ZH,
            input_variables=["format_instructions"],
        )

        system_str = system_prompt.format(
            format_instructions=parser.get_format_instructions(),
        )

        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=system_str),
             ('human', ASK_USER_ZH)]
        )

    if self_query:
        if expr_stmt is not None:
            retriever = expr_retriever(vec_store, doc_store, reranker, expr_stmt)
        else:
            retriever = self_query_retriever(vec_store, doc_store, reranker)
    else:

        retriever = base_retriever(vec_store, doc_store, reranker)

    formatter = itemgetter("docs") | RunnableLambda(format_docs)

    chain = prompt | llm | parser
    answer_chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        .assign(context=formatter)
        .assign(answer=chain)
        .pick(["answer", "docs"])
    )

    result = answer_chain.invoke(question)

    return result
