import os

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.milvus import Milvus
from loguru import logger
from Config import config
from utils.TimeUtil import timer

logger.add('log/runtime.log')


def assemble_md():
    for root, dirs, files in os.walk(config.XML_OUTPUT):
        for file in files:
            file_path = os.path.join(root, file)
            file_year = os.path.basename(root)
            doi = file.replace('.grobid.tei.xml', '')
            logger.info(f'loading <{doi}> ({file_year})...')

            from utils.GrobidUtil import parse_xml, save_to_md

            data = parse_xml(f'{file_path}')

            md_path = os.path.join(config.MD_OUTPUT, f'{file_year}/{doi}.md')
            if os.path.exists(md_path):
                save_to_md(data, md_path, True)
            else:
                save_to_md(data, md_path, False)
                logger.warning(f'markdown not find!')


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

    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vector_db = Milvus(
        embedding,
        collection_name=config.milvus_config.COLLECTION_NAME,
        connection_args={
            'host': config.milvus_config.MILVUS_HOST,
            'port': config.milvus_config.MILVUS_PORT
        },
    )

    logger.info('done')

    logger.info('start loading file...')

    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_year = os.path.basename(root)
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
    if not os.path.exists(config.MD_OUTPUT):
        os.makedirs(config.MD_OUTPUT)
    if not os.path.exists(config.XML_OUTPUT):
        os.makedirs(config.XML_OUTPUT)

    if config.PDF_PARSER == 'grobid':
        from utils.GrobidUtil import parse_pdf

        parse_pdf(config.PDF_ROOT)
        assemble_md()

    load_md(config.MD_OUTPUT)
