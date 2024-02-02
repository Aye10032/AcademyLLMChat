import os

from Config import config
from loguru import logger
from utils.GrobidUtil import parse_pdf, save_to_md, parse_xml

logger.add('log/pdf2md.log')


def assemble_md():
    for root, dirs, files in os.walk(config.XML_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            file_year = os.path.basename(root)
            doi = file.replace('.grobid.tei.xml', '')
            logger.info(f'loading <{doi}> ({file_year})...')

            data = parse_xml(f'{file_path}')

            md_path = os.path.join(config.MD_PATH, f'{file_year}/{doi}.md')
            if os.path.exists(md_path):
                save_to_md(data, md_path, True)
            else:
                save_to_md(data, md_path, False)
                logger.warning(f'markdown not find!')


if __name__ == '__main__':
    parse_pdf(config.PDF_PATH, config.MD_PATH)
    logger.info('PDF解析为markdown完成，开始处理markdown文件...')
    assemble_md()
