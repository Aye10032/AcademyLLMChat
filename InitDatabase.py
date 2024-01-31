import os

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.milvus import Milvus
from loguru import logger
from Config import config
from utils.TimeUtil import timer

logger.add('log/runtime.log')


@timer
def load_md(base_path):
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[('#', 'Title'), ('##', 'SubTitle'), ('###', 'Title3')]
    )
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separators=['\n\n', '\n']
    )

    logger.info('start building vector database...')
    milvus_cfg = config.milvus_config

    embedding = HuggingFaceBgeEmbeddings(
        model_name=milvus_cfg.get_model(),
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_db = Milvus(
        embedding,
        collection_name=milvus_cfg.get_collection()['collection_name'],
        connection_args={
            'host': milvus_cfg.MILVUS_HOST,
            'port': milvus_cfg.MILVUS_PORT
        },
        drop_old=True
    )

    logger.info('done')

    logger.info('start loading file...')

    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_year = int(os.path.basename(root))
            doi = file.replace('@', '/').replace('.md', '')
            logger.info(f'loading <{file}> ({file_year}) {file_path}...')

            with open(file_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
            head_split_docs = md_splitter.split_text(md_text)
            for i, doc in enumerate(head_split_docs):
                doc.metadata['doi'] = doi
                doc.metadata['year'] = file_year
            md_docs = r_splitter.split_documents(head_split_docs)

            vector_db.add_documents(md_docs)
    logger.info(f'done')


if __name__ == '__main__':
    load_md(config.MD_OUTPUT)
