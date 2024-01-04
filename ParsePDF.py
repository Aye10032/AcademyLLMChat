from config.Config import PDF_ROOT, MD_OUTPUT
from utils.NougatUtil import parse_pdf_to_md
from loguru import logger

logger.add('log/pdf2md.log')

if __name__ == '__main__':
    parse_pdf_to_md(PDF_ROOT, MD_OUTPUT)
    logger.info('PDF解析为markdown完成，开始处理markdown文件...')

