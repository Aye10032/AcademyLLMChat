from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus

md_path = '../preprocess/output/markdown/11/10.1016@j.febslet.2011.05.015.grobid.tei.md'


md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[('#', 'Title'), ('##', 'SubTitle'), ('###', 'Title3')])
with open(md_path, 'r') as f:
    md_text = f.read()
md_docs = md_splitter.split_text(md_text)
for i, doc in enumerate(md_docs):
    doc.metadata['doi'] = '10.1016/j.febslet.2011.05.015'

model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vector_db = Milvus.from_documents(
    md_docs,
    embedding,
    collection_name="LLM_test_1",
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

