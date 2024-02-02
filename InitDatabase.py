import argparse
import os

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.milvus import Milvus
from loguru import logger
from tqdm import tqdm

from Config import config
from utils.TimeUtil import timer

logger.add('log/init_database.log')


@timer
def load_md(base_path):
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[('#', 'Title'), ('##', 'Section'), ('###', 'Subtitle'), ('####', 'Subtitle')]
    )
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separators=['\n\n', '\n']
    )

    logger.info('start building vector database...')
    milvus_cfg = config.milvus_config

    model = milvus_cfg.get_model()
    collection = milvus_cfg.get_collection()['collection_name']

    logger.info(f'load collection [{collection}], using model {model}')

    embedding = HuggingFaceBgeEmbeddings(
        model_name=model,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_db = Milvus(
        embedding,
        collection_name=collection,
        connection_args={
            'host': milvus_cfg.MILVUS_HOST,
            'port': milvus_cfg.MILVUS_PORT
        },
        drop_old=True
    )

    logger.info('done')

    logger.info('start loading file...')

    for root, dirs, files in os.walk(base_path):
        for file in tqdm(files, total=len(files)):
            file_path = os.path.join(root, file)
            file_year = int(os.path.basename(root))
            doi = file.replace('@', '/').replace('.md', '')

            with open(file_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
            head_split_docs = md_splitter.split_text(md_text)
            for i, doc in enumerate(head_split_docs):
                doc.metadata['doi'] = doi
                doc.metadata['year'] = file_year
            md_docs = r_splitter.split_documents(head_split_docs)

            # logger.info(f'loading <{file}> ({file_year})...')
            try:
                vector_db.add_documents(md_docs)
            except Exception as e:
                logger.error(f'loading <{file}> ({file_year}) fail')
                logger.error(e)
    logger.info(f'done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', '-C', type=int, help='初始化特定collection，从0开始')
    args = parser.parse_args()
    if args.collection:
        if args.collection >= len(config.milvus_config.COLLECTIONS):
            logger.error(f'collection index {args.collection} out of range')
            exit(1)
        else:
            logger.info(f'Only init collection {args.collection}')
            config.set_collection(args.collection)
            load_md(config.MD_PATH)
    else:
        for i in range(len(config.milvus_config.COLLECTIONS)):
            logger.info(f'Start init collection {i}')
            # config.set_collection(i)
            # load_md(config.MD_PATH)
