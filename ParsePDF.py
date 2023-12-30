from utils.PdfUtil import parse_pdf_to_md
from loguru import logger

logger.add('log/pdf2md.log')

parse_pdf_to_md('documents', 'output')
