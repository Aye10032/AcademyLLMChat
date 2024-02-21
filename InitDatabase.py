import json
import os
import shutil
import sys

import yaml
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores.milvus import Milvus
from loguru import logger
from tqdm import tqdm

from utils.TimeUtil import timer

logger.remove()
handler_id = logger.add(sys.stderr, level="INFO")
logger.add('log/init_database.log')


def init_retriever() -> ParentDocumentRetriever:
    logger.info('start building vector database...')
    milvus_cfg = config.milvus_config

    collection = milvus_cfg.get_collection().NAME

    if milvus_cfg.get_collection().LANGUAGE == 'zh':
        model = config.milvus_config.ZH_MODEL

        embedding = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        model = config.milvus_config.EN_MODEL

        embedding = HuggingFaceBgeEmbeddings(
            model_name=model,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
    logger.info(f'load collection [{collection}], using model {model}')

    doc_store = SqliteDocStore(
        connection_string=config.get_sqlite_path()
    )

    if milvus_cfg.USING_REMOTE:
        connection_args = {
            'uri': milvus_cfg.REMOTE_DATABASE['url'],
            'user': milvus_cfg.REMOTE_DATABASE['username'],
            'password': milvus_cfg.REMOTE_DATABASE['password'],
            'secure': True,
        }
    else:
        connection_args = {
            'host': milvus_cfg.MILVUS_HOST,
            'port': milvus_cfg.MILVUS_PORT,
        }

    vector_db = Milvus(
        embedding,
        collection_name=collection,
        connection_args=connection_args,
        index_params=milvus_cfg.get_collection().INDEX_PARAM,
        drop_old=True
    )

    logger.info('done')

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
        vectorstore=vector_db,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever


@timer
def load_md(base_path):
    retriever = init_retriever()
    logger.info('start loading file...')

    for root, dirs, files in os.walk(base_path):
        if len(files) == 0:
            continue

        year = os.path.basename(root)
        for _file in tqdm(files, total=len(files), desc=f'load file in ({year})'):
            if _file.endswith('.grobid.tei.xml'):
                file_path = os.path.join(config.get_md_path(), year, _file.replace('.grobid.tei.xml', '.md'))

                if not os.path.exists(file_path):
                    logger.warning(f'loading <{_file}> ({year}) fail')
                    continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_text = f.read()

                md_docs = split_markdown_text(md_text)

                try:
                    retriever.add_documents(md_docs)
                except Exception as e:
                    logger.error(f'loading <{_file}> ({year}) fail')
                    logger.error(e)
            else:
                file_path = os.path.join(root, _file)
                doi = _file.replace('@', '/').replace('.xml', '')

                with open(file_path, 'r', encoding='utf-8') as f:
                    xml_text = f.read()

                data = parse_paper_data(xml_text, year, doi)

                if not data['norm']:
                    continue

                docs = section_to_documents(data['sections'], data['author'], int(year), doi)

                try:
                    retriever.add_documents(docs)
                except Exception as e:
                    logger.error(f'loading <{_file}> ({year}) fail')
                    logger.error(e)

    logger.info(f'done')


@timer
def load_xml(base_path):
    retriever = init_retriever()
    logger.info('start loading file...')

    for root, dirs, files in os.walk(base_path):
        if len(files) == 0:
            continue

        year = os.path.basename(root)
        for _file in tqdm(files, total=len(files), desc=f'load file in ({year})'):
            file_path = os.path.join(root, _file)
            doi = _file.replace('@', '/').replace('.xml', '')

            with open(file_path, 'r', encoding='utf-8') as f:
                xml_text = f.read()

            data = parse_paper_data(xml_text, year, doi)

            if not data['norm']:
                continue

            output_path = os.path.join(config.get_md_path(), year, doi.replace('/', '@') + '.md')
            os.makedirs(os.path.join(config.get_md_path(), year), exist_ok=True)
            save_to_md(data['sections'], output_path)

            docs = section_to_documents(data['sections'], data['author'], int(year), doi)

            try:
                retriever.add_documents(docs)
            except Exception as e:
                logger.error(f'loading <{_file}> ({year}) fail')
                logger.error(e)

    logger.info(f'done')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='init vector database')
    parser.add_argument(
        '--collection',
        '-C',
        nargs='?',
        type=int,
        help='Initialize a specific collection, starting from 0.'
    )
    parser.add_argument(
        '--auto_create',
        '-A',
        action='store_true',
        help='Automatic database initialization based on directory structure'
    )
    parser.add_argument(
        '--force',
        '-F',
        action='store_true',
        help='Force override of existing configurations'
    )
    parser.add_argument(
        '--build_reference',
        '-R',
        action='store_true',
        help='Building a reference tree when creating a document'
    )
    args = parser.parse_args()

    if args.auto_create:
        yml_path = 'config.yml'
        if not os.path.exists(yml_path):
            logger.info('config dose not exits')
            shutil.copy('config.example.yml', yml_path)

        with open(file=yml_path, mode='r', encoding='utf-8') as file:
            yml = yaml.load(file, Loader=yaml.FullLoader)

        DATA_ROOT = yml['data_root']
        cfg_path = os.path.join(DATA_ROOT, 'collections.json')

        if not args.force and os.path.exists(cfg_path):
            logger.info('config file exists, use existing config file')
        else:
            cols = [{"collection_name": collection,
                     "language": 'en',
                     "title": collection,
                     "description": f'This is a collection about {collection}',
                     "index_param": {
                         "metric_type": 'L2',
                         "index_type": 'HNSW',
                         "params": {"M": 8, "efConstruction": 64},
                     }}
                    for collection in os.listdir(DATA_ROOT)
                    if os.path.isdir(os.path.join(DATA_ROOT, collection))]

            json.dump({"collections": cols}, open(cfg_path, 'w', encoding='utf-8'))
            logger.info(f'auto create config file {cfg_path}')

    from Config import config

    from llm.storage.SqliteStore import SqliteDocStore
    from utils.FileUtil import save_to_md, section_to_documents
    from utils.MarkdownPraser import split_markdown_text
    from utils.PMCUtil import parse_paper_data

    if args.collection is not None:
        if not args.collection:
            for i in range(len(config.milvus_config.COLLECTIONS)):
                logger.info(f'Start init collection {i}')
                config.set_collection(i)
                load_md(config.get_xml_path())
        elif args.collection is True:
            if args.collection >= len(config.milvus_config.COLLECTIONS) or args.collection < 0:
                logger.error(f'collection index {args.collection} out of range')
                exit(1)
            else:
                config.set_collection(args.collection)
                if args.build_reference:
                    logger.info(f'Only init collection {args.collection} with reference')
                    load_xml(config.get_xml_path())
                else:
                    logger.info(f'Only init collection {args.collection}')
                    load_md(config.get_md_path())
