import os

from tqdm import tqdm

from Config import Config
from loguru import logger
from utils.GrobidUtil import parse_xml, GrobidConnector
from utils.MarkdownPraser import save_to_md

logger.add('log/pdf2md.log')


def _pdf2xml(input_path: str | bytes):
    output_path = os.path.join(input_path, 'xml')
    with GrobidConnector(gr_cfg) as connector:
        connector.parse_files(input_path, output_path, skip_exist=True)


def _xml2md(input_path: str | bytes):
    xml_path = os.path.join(input_path, 'xml')
    md_path = os.path.join(input_path, 'md')

    for root, _, files in os.walk(xml_path):
        if not files:
            continue

        for file in tqdm(files, total=len(files)):
            file_path = os.path.join(root, file)

            data = parse_xml(file_path)
            year = data.info.year
            doi = data.info.doi

            filename = f"{doi.replace('/', '@')}.md" if doi else file.replace('.xml', '.md')
            year_folder = str(year) if year else 'unknown'
            md_file = os.path.join(md_path, year_folder, filename)

            os.makedirs(os.path.dirname(md_file), exist_ok=True)
            save_to_md(data, md_file)


def _assemble_md(silent: bool = True):
    collection = config.milvus_config.get_collection().collection_name
    for root, dirs, files in os.walk(config.get_xml_path(collection)):
        if len(files) == 0:
            continue

        file_year = os.path.basename(root)
        for file in tqdm(files, total=len(files), desc=f'{file_year}'):
            if not file.endswith('.grobid.tei.xml'):
                if not silent:
                    logger.warning(f'skip {file}')
                continue

            file_path = os.path.join(root, file)
            doi = file.replace('.grobid.tei.xml', '')

            data = parse_xml(file_path)

            md_path = os.path.join(config.get_md_path(collection), file_year, f'{doi}.md')
            os.makedirs(os.path.join(config.get_md_path(collection), file_year), exist_ok=True)
            save_to_md(data, md_path)


if __name__ == '__main__':
    config = Config()
    gr_cfg = config.grobid_config
    pdf_path = 'D:/program/DATA/raman/拉曼文献库1'

    logger.info('开始多线程解析PDF文件...')
    _pdf2xml(pdf_path)
    logger.info('PDF解析为xml完成，开始处理xml文件...')
    _xml2md(pdf_path)
