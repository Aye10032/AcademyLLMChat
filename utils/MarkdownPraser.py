from io import StringIO

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile


def split_markdown(document: UploadedFile, year: int):
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[('#', 'Title'), ('##', 'Section'), ('###', 'Subtitle'), ('####', 'Subtitle')]
    )
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separators=['\n\n', '\n']
    )

    doi = document.name.replace('@', '/').replace('.md', '')

    stringio = StringIO(document.getvalue().decode("utf-8"))
    string_data = stringio.read()
    head_split_docs = md_splitter.split_text(string_data)
    for doc in head_split_docs:
        doc.metadata['doi'] = doi
        doc.metadata['year'] = year
    md_docs = r_splitter.split_documents(head_split_docs)

    return md_docs
