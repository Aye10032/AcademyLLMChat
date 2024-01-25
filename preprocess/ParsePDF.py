from Config import config
from loguru import logger

logger.add('log/pdf2md.log')

if __name__ == '__main__':
    if config.PDF_PARSER == 'grobid':
        from utils import parse_pdf

        pass
    elif config.PDF_PARSER == 'nougat':
        from utils import parse_pdf

        parse_pdf(config.PDF_ROOT, config.MD_OUTPUT)
        logger.info('PDF解析为markdown完成，开始处理markdown文件...')
