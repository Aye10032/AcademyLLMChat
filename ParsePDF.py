from config.Config import PDF_ROOT, MMD_OUTPUT
from utils.PdfUtil import parse_pdf_to_md
from loguru import logger

logger.add('log/pdf2md.log')

if __name__ == '__main__':
    parse_pdf_to_md(PDF_ROOT, MMD_OUTPUT)
