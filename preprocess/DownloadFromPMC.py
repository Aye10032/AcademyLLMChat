import os
import random
import sys
import time

import pandas as pd
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.milvus import Milvus
from loguru import logger
from tqdm import tqdm

from Config import config
from llm.storage.SqliteStore import SqliteDocStore
from utils.FileUtil import save_to_md, section_to_documents
from utils.PMCUtil import download_paper_data, parse_paper_data

logger.remove()
handler_id = logger.add(sys.stderr, level="INFO")
logger.add('log/pmc.log')


def init_retriever() -> ParentDocumentRetriever:
    doc_store = SqliteDocStore(
        connection_string=config.get_sqlite_path()
    )

    milvus_cfg = config.milvus_config
    embedding = HuggingFaceBgeEmbeddings(
        model_name=milvus_cfg.EN_MODEL,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
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
        collection_name=milvus_cfg.get_collection().NAME,
        connection_args=connection_args,
        index_params=milvus_cfg.get_collection().INDEX_PARAM
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separators=['\n\n', '\n'],
        keep_separator=False
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
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


def download_from_pmc(csv_file: str):
    df = pd.read_csv(csv_file, encoding='utf-8', dtype={'title': 'str', 'pmc_id': 'str', 'doi': 'str', 'year': 'str'})

    df_output = df.copy()
    length = df_output.shape[0]
    for index, row in tqdm(df_output.iterrows(), total=length, desc='download xml'):
        if not pd.isna(row.year):
            continue

        pmcid = row.pmc_id
        data = download_paper_data(pmcid)

        df_output.at[index, 'doi'] = data['doi']
        df_output.at[index, 'year'] = data['year']
        df_output.to_csv(csv_file, index=False, encoding='utf-8')
        time.sleep(random.uniform(2.0, 5.0))


def solve_xml(csv_file: str):
    retriever = init_retriever()
    df = pd.read_csv(csv_file, encoding='utf-8', dtype={'title': 'str', 'pmc_id': 'str', 'doi': 'str', 'year': 'str'})

    df_output = df.copy()
    length = df_output.shape[0]
    for index, row in tqdm(df_output.iterrows(), total=length, desc='adding documents'):
        if not pd.isna(row.title):
            continue

        year: str = row.year
        doi: str = row.doi
        with open(os.path.join(config.get_xml_path(), year, doi.replace('/', '@') + '.xml'), 'r',
                  encoding='utf-8') as f:
            xml_text = f.read()
        data = parse_paper_data(xml_text, year, doi)

        if not data['norm']:
            df_output.at[index, 'title'] = data['title']
            df_output.to_csv(csv_file, index=False, encoding='utf-8')
            continue

        output_path = os.path.join(config.get_md_path(), year, doi.replace('/', '@') + '.md')
        os.makedirs(os.path.join(config.get_md_path(), year), exist_ok=True)
        save_to_md(data['sections'], output_path)

        docs = section_to_documents(data['sections'], data['author'], int(year), doi)
        retriever.add_documents(docs)

        df_output.at[index, 'title'] = data['title']
        df_output.to_csv(csv_file, index=False, encoding='utf-8')


if __name__ == '__main__':
    # term = 'Raman[Title] AND ("2019/02/01"[PDat] : "2024/01/30"[PDat])&retmode=json&retmax=2000'
    # get_pmc_id(term)

    config.set_collection(1)
    # download_from_pmc('pmlist.csv')
    solve_xml('pmlist.csv')
