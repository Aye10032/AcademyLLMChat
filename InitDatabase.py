import os

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.milvus import Milvus
from loguru import logger
from Config import config
from utils.TimeUtil import timer

logger.add('log/runtime.log')


@timer
def load_md(base_path):
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[('#', 'Title'), ('##', 'SubTitle'), ('###', 'Title3')])

    md_docs = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_year = os.path.basename(root)
            logger.info(f'loading <{file}> ({file_year}) {file_path}...')

            with open(file_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
            md_doc = md_splitter.split_text(md_text)
            for i, doc in enumerate(md_doc):
                doc.metadata['doi'] = file.replace('@', '/').replace('.md', '')
                doc.metadata['year'] = file_year
                md_docs.append(doc)
    logger.info(f'loaded {len(md_docs)}')

    logger.info('start building vector database...')

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
        collection_name=config.milvus_config.COLLECTION_NAME,
        connection_args={
            'host': config.milvus_config.MILVUS_HOST,
            'port': config.milvus_config.MILVUS_PORT
        },
    )

    logger.info('done')


if __name__ == '__main__':
    load_md(config.MD_OUTPUT)
