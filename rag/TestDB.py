import httpx
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus, milvus
from langchain_core.vectorstores import VectorStoreRetriever
from Config import config

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

question = 'What do you know about the historical research on microalgae oil production?'

http_client = httpx.Client(proxies="http://127.0.0.1:11451")
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k",
                 http_client=http_client,
                 temperature=0,
                 openai_api_key=config.openai_config.API_KEY)
retriever = VectorStoreRetriever(vectorstore=vector_db)
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={'prompt': QA_CHAIN_PROMPT}
)

result = qa_chain({'query': question})
print(result['result'])
