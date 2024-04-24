import os

from tqdm import tqdm

from Config import Config
from loguru import logger
from utils.GrobidUtil import parse_pdf, save_to_md, parse_xml

logger.add('log/pdf2md.log')


def assemble_md(silent: bool = True):
    for root, dirs, files in os.walk(config.get_xml_path()):
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

            md_path = os.path.join(config.get_md_path(), file_year, f'{doi}.md')
            os.makedirs(os.path.join(config.get_md_path(), file_year), exist_ok=True)
            save_to_md(data, md_path)


if __name__ == '__main__':
    config = Config()
    # parse_pdf(config.get_pdf_path(), config)
    logger.info('PDF解析为xml完成，开始处理xml文件...')
    assemble_md()
