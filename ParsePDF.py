from Config import PDF_ROOT, MD_OUTPUT, PDF_PARSER
from loguru import logger

logger.add('log/pdf2md.log')

if __name__ == '__main__':
    if PDF_PARSER == 'grobid':
        from preprocess.utils.GrobidUtil import parse_pdf

        pass
    elif PDF_PARSER == 'nougat':
        from preprocess.utils.NougatUtil import parse_pdf

        parse_pdf(PDF_ROOT, MD_OUTPUT)
        logger.info('PDF解析为markdown完成，开始处理markdown文件...')
