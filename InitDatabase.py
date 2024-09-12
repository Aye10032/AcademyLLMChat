import json
import os
import shutil
import sys
from datetime import datetime

import yaml
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger
from tqdm import tqdm

from llm.EmbeddingCore import BgeM3Embeddings
from utils.Decorator import timer
from utils.entities.UserProfile import User, UserGroup

logger.remove()
handler_id = logger.add(sys.stderr, level="INFO")
logger.add('log/init_database.log')


def init_retriever() -> ParentDocumentRetriever:
    logger.info('start building vector database...')
    milvus_cfg = config.milvus_config

    collection_name = milvus_cfg.get_collection().collection_name
    embed_cfg = config.embedding_config
    embedding = BgeM3Embeddings(
        model_name=embed_cfg.model,
        model_kwargs={
            'device': 'cuda',
            'normalize_embeddings': embed_cfg.normalize,
            'use_fp16': embed_cfg.fp16
        },
        local_load=embed_cfg.save_local,
        local_path=embed_cfg.local_path
    )
    logger.info(f'load collection [{collection_name}], using model {embed_cfg.model}')

    if args.drop_old:
        doc_store = SqliteDocStore(
            connection_string=config.get_sqlite_path(collection_name),
            drop_old=True
        )
    else:
        doc_store = SqliteDocStore(
            connection_string=config.get_sqlite_path(collection_name)
        )

    vector_db = Milvus(
        embedding,
        collection_name=collection_name,
        connection_args=milvus_cfg.get_conn_args(),
        index_params=milvus_cfg.get_collection().index_param,
        drop_old=True,
        auto_id=True
    )

    logger.info('done')

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separators=['\n\n', '\n'],
        keep_separator=False
    )

    if milvus_cfg.get_collection().language == 'en':
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
            separators=['.', '\n\n', '\n'],
            keep_separator=False
        )
    elif milvus_cfg.get_collection().language == 'zh':
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
            separators=['。', '？', '\n\n', '\n'],
            keep_separator=False
        )
    else:
        raise Exception(f'error language {milvus_cfg.get_collection().language}')

    retriever = ParentDocumentRetriever(
        vectorstore=vector_db,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever


@timer
def load_md(base_path: str) -> None:
    """
    加载markdown文件到检索器中。

    :param base_path: 基础路径，包含年份子目录，每个子目录下包含markdown和xml文件。
    :return: 无返回值
    """
    # 初始化检索器，并添加初始文档

    retriever = init_retriever()
    now_collection = config.milvus_config.get_collection().collection_name
    init_doc = Document(page_content=f'This is a collection about {now_collection}',
                        metadata={
                            'title': 'About this collection',
                            'section': 'Abstract',
                            'author': 'administrator',
                            'year': datetime.now().year,
                            'type': -1,
                            'keywords': 'collection',
                            'doi': ''
                        })
    retriever.add_documents([init_doc])
    logger.info('start loading file...')

    # 遍历基础路径下的所有文件和子目录
    for root, dirs, files in os.walk(base_path):
        # 跳过空目录
        if len(files) == 0:
            continue

        # 提取年份信息
        year = os.path.basename(root)
        for _file in tqdm(files, total=len(files), desc=f'load file in ({year})'):
            # 加载并处理markdown文件
            file_path = os.path.join(config.get_md_path(now_collection), year, _file)

            # 分割markdown文本为多个文档
            md_docs, reference_data = load_from_md(file_path)

            # 尝试将分割得到的文档添加到检索器
            try:
                retriever.add_documents(md_docs)
                with ReferenceStore(config.get_reference_path()) as ref_store:
                    ref_store.add_reference(reference_data)
            except Exception as e:
                logger.error(f'loading <{_file}> ({year}) fail')
                logger.error(e)

    logger.info(f'done')


def create_userdb():
    connect_str = config.get_user_db()
    os.makedirs(os.path.dirname(connect_str))

    with ProfileStore(
            connection_string=connect_str
    ) as profile_store:
        init_username = config.yml['user_login_config']['admin_user']['username']
        init_password = config.yml['user_login_config']['admin_user']['password']
        admin = User(
            name=init_username,
            password=init_password,
            user_group=UserGroup.ADMIN.value
        )
        profile_store.create_user(admin)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='init vector database')
    parser.add_argument(
        '--collection',
        '-C',
        nargs='?',
        const=-1,
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
        '--drop_old',
        '-D',
        action='store_true',
        help='Whether to delete the original reference database'
    )
    parser.add_argument(
        '--user',
        '-U',
        action='store_true',
        help='Initialize user-related databases, '
             'including creating SQLite database files that hold user information and initializing administrator user accounts.'
    )
    args = parser.parse_args()

    if args.auto_create:
        yml_path = 'config.yml'
        if not os.path.exists(yml_path):
            logger.info('config dose not exits')
            shutil.copy('config.example.yml', yml_path)

        with open(file=yml_path, mode='r', encoding='utf-8') as file:
            yml = yaml.load(file, Loader=yaml.FullLoader)

        DATA_ROOT = yml['paper_directory']['data_root']
        cfg_path = os.path.join(DATA_ROOT, 'collections.json')

        if not args.force and os.path.exists(cfg_path):
            logger.info('config file exists, use existing config file')
        else:
            cols = [
                {
                    "collection_name": collection,
                    "language": 'en',
                    "title": collection,
                    "description": f'This is a collection about {collection}',
                    "index_param": {
                        "metric_type": 'L2',
                        "index_type": 'HNSW',
                        "params": {"M": 8, "efConstruction": 64},
                    },
                    "visitor_visible": True,
                }
                for collection in os.listdir(DATA_ROOT)
                if os.path.isdir(os.path.join(DATA_ROOT, collection))
            ]

            json.dump({"collections": cols}, open(cfg_path, 'w', encoding='utf-8'))
            logger.info(f'auto create config file {cfg_path}')

    from Config import Config

    config = Config()

    from storage.SqliteStore import SqliteDocStore, ReferenceStore, ProfileStore
    from utils.MarkdownPraser import load_from_md

    if args.drop_old:
        with ReferenceStore(config.get_reference_path()) as _store:
            _store.drop_old()
            logger.info('drop old database.')

    if args.collection is not None:
        if args.collection == -1:
            for i in range(len(config.milvus_config.collections)):
                logger.info(f'Start init collection {i}')
                config.set_collection(i)
                load_md(config.get_md_path(config.milvus_config.get_collection().collection_name))
        else:
            if args.collection >= len(config.milvus_config.collections) or args.collection < -1:
                logger.error(f'collection index {args.collection} out of range')
                exit(1)
            else:
                config.set_collection(args.collection)
                logger.info(f'Only init collection {args.collection}')
                load_md(config.get_md_path(config.milvus_config.get_collection().collection_name))

    if args.user:
        logger.info('Create admin profile...')
        create_userdb()
