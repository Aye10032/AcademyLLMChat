import json
import os
import shutil
import sys
from datetime import datetime

import yaml
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.documents import Document
from loguru import logger
from tqdm import tqdm

from utils.Decorator import timer

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

        embedding = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
    logger.info(f'load collection [{collection}], using model {model}')

    doc_store = SqliteDocStore(
        connection_string=config.get_sqlite_path(),
        drop_old=True
    )

    vector_db = Milvus(
        embedding,
        collection_name=collection,
        connection_args=milvus_cfg.get_conn_args(),
        index_params=milvus_cfg.get_collection().INDEX_PARAM,
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
def load_md(base_path) -> None:
    """
    加载markdown文件到检索器中。

    :param base_path: 基础路径，包含年份子目录，每个子目录下包含markdown和xml文件。
    :return: 无返回值
    """
    # 初始化检索器，并添加初始文档
    retriever = init_retriever()
    init_doc = Document(page_content=f'This is a collection about {config.milvus_config.get_collection().NAME}',
                        metadata={
                            'title': 'About this collection',
                            'section': 'Abstract',
                            'author': '',
                            'doi': '',
                            'year': datetime.now().year,
                            'ref': ''
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
            if _file.endswith('.grobid.tei.xml'):
                file_path = os.path.join(config.get_md_path(), year, _file.replace('.grobid.tei.xml', '.md'))

                # 如果markdown文件不存在，则跳过
                if not os.path.exists(file_path):
                    logger.warning(f'loading <{_file}> ({year}) fail')
                    continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_text = f.read()

                # 分割markdown文本为多个文档
                md_docs = split_markdown_text(md_text)

                # 尝试将分割得到的文档添加到检索器
                try:
                    retriever.add_documents(md_docs)
                except Exception as e:
                    logger.error(f'loading <{_file}> ({year}) fail')
                    logger.error(e)
            # 处理xml文件，将其转换为文档并添加到检索器
            else:
                file_path = os.path.join(root, _file)
                doi = _file.replace('@', '/').replace('.xml', '')

                with open(file_path, 'r', encoding='utf-8') as f:
                    xml_text = f.read()

                # 解析xml文件内容
                data = parse_paper_data(xml_text, year, doi)

                # 如果数据未规范化，则跳过
                if not data['norm']:
                    continue

                # 将解析得到的数据转换为文档
                docs = section_to_documents(data['sections'], data['author'], int(year), doi)

                # 尝试将转换得到的文档添加到检索器
                try:
                    retriever.add_documents(docs)
                except Exception as e:
                    logger.error(f'loading <{_file}> ({year}) fail')
                    logger.error(e)

    logger.info(f'done')


@timer
def load_xml(base_path) -> None:
    """
    加载XML文件并将其转换为Markdown格式，同时将相关信息添加到检索器中。

    :param base_path: 基础路径，包含所有需要加载的XML文件的目录。
    :return: 无返回值。
    """
    # 初始化文档检索器，并添加一个初始文档
    retriever = init_retriever()
    init_doc = Document(page_content=f'This is a collection about {config.milvus_config.get_collection().NAME}',
                        metadata={
                            'title': 'About this collection',
                            'section': 'Abstract',
                            'author': '',
                            'doi': '',
                            'year': datetime.now().year,
                            'ref': ''
                        })
    retriever.add_documents([init_doc])

    # 开始加载文件的日志记录
    logger.info('start loading file...')

    # 遍历基础路径下的所有目录和文件
    for root, dirs, files in os.walk(base_path):
        # 如果目录中没有文件，则跳过
        if len(files) == 0:
            continue

        # 从目录路径中提取年份信息
        year = os.path.basename(root)
        for _file in tqdm(files, total=len(files), desc=f'load file in ({year})'):
            file_path = os.path.join(root, _file)
            doi = _file.replace('@', '/').replace('.xml', '')

            # 读取XML文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                xml_text = f.read()

            # 解析XML文件数据
            data = parse_paper_data(xml_text, year, doi)

            # 如果规范化的数据为空，则跳过当前文件
            if not data['norm']:
                continue

            # 将解析后的数据保存为Markdown格式
            output_path = os.path.join(config.get_md_path(), year, doi.replace('/', '@') + '.md')
            os.makedirs(os.path.join(config.get_md_path(), year), exist_ok=True)
            save_to_md(data['sections'], output_path)

            # 将文章章节数据转换为文档格式，用于检索器
            docs = section_to_documents(data['sections'], data['author'], int(year), doi)

            try:
                # 将文档添加到检索器中
                retriever.add_documents(docs)
            except Exception as e:
                # 记录添加文档失败的日志
                logger.error(f'loading <{_file}> ({year}) fail')
                logger.error(e)

    # 完成加载的日志记录
    logger.info(f'done')


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

    from Config import Config

    config = Config()

    from llm.storage.SqliteStore import SqliteDocStore
    from utils.FileUtil import save_to_md, section_to_documents
    from utils.MarkdownPraser import split_markdown_text
    from utils.PMCUtil import parse_paper_data

    if args.collection is not None:
        if args.collection == -1:
            for i in range(len(config.milvus_config.COLLECTIONS)):
                logger.info(f'Start init collection {i}')
                config.set_collection(i)
                load_md(config.get_xml_path())
        else:
            if args.collection >= len(config.milvus_config.COLLECTIONS) or args.collection < -1:
                logger.error(f'collection index {args.collection} out of range')
                exit(1)
            else:
                config.set_collection(args.collection)
                if args.build_reference:
                    logger.info(f'Only init collection {args.collection} with reference')
                    load_xml(config.get_xml_path())
                else:
                    logger.info(f'Only init collection {args.collection}')
                    load_md(config.get_xml_path())
