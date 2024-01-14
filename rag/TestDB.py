from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus

model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vector_db = Milvus(
    embedding,
    collection_name="LLM_test_1",
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

question = 'what is NPQ?'

docs = vector_db.similarity_search(question, k=2)
