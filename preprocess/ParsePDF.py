from Config import config
from loguru import logger
from utils.GrobidUtil import parse_pdf

logger.add('log/pdf2md.log')

if __name__ == '__main__':
    parse_pdf(config.PDF_ROOT, config.MD_OUTPUT)
    logger.info('PDF解析为markdown完成，开始处理markdown文件...')
